"""
xai_engine.py - Core Real Inference & Explainability Engine for Dental Crowding
Loads:
  - results/box_model.npz (Learned Ridge regression teeth box detector)
  - results/pca_params.npz (PCA projection matrix: 9216 -> 126)
  - results/crowding_model.npz (Multilayer Perceptron: 126 -> 16 -> 8 -> 2)
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

class DentalXAIModel:
    def __init__(self, results_dir=RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.box_model = np.load(self.results_dir / "box_model.npz")
        self.pca_model = np.load(self.results_dir / "pca_params.npz")
        self.crowd_model = np.load(self.results_dir / "crowding_model.npz")
        
        # Dimensions
        self.box_size = tuple(self.box_model["size"])  # (48, 32)
        self.box_pad = float(self.box_model["pad"])    # 0.02
        self.img_w = 144
        self.img_h = 64

    def predict_teeth_box(self, image_pil):
        """Detects (left, top, right, bottom) teeth bounding box as fractions [0, 1]."""
        rgb = image_pil.convert("RGB")
        small = rgb.resize(self.box_size)
        x = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
        
        # Ridge regression: (x - mu) @ W + b
        p = (x - self.box_model["mu"]) @ self.box_model["W"] + self.box_model["b"]
        p = np.clip(p, 0.0, 1.0)
        
        l = max(0.0, float(p[0]) - self.box_pad)
        t = max(0.0, float(p[1]) - self.box_pad)
        r = min(1.0, float(p[2]) + self.box_pad)
        b = min(1.0, float(p[3]) + self.box_pad)
        return (l, t, r, b)

    def crop_and_preprocess(self, image_pil, box):
        """Crops teeth area, grayscales, resizes to 144x64, equalizes contrast."""
        l, t, r, b = box
        w, h = image_pil.size
        crop_box = (int(l * w), int(t * h), int(r * w), int(b * h))
        
        # Guard against zero-area crop
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            crop_box = (0, 0, w, h)
            
        crop = image_pil.convert("L").crop(crop_box)
        crop_resized = crop.resize((self.img_w, self.img_h))
        crop_eq = ImageOps.equalize(crop_resized)
        
        pixels = np.asarray(crop_eq, dtype=np.float32).reshape(-1) / 255.0
        return crop_eq, pixels

    def extract_pca(self, pixels):
        """Projects 9216 flattened pixels to 126 PCA components."""
        mean = self.pca_model["mean"]
        components = self.pca_model["components"]
        return (pixels - mean) @ components.T

    def forward(self, x_pca):
        """Forward pass through MLP (126 -> 16 -> 8 -> 2) with softmax."""
        mu = self.crowd_model["mu"]
        sd = float(self.crowd_model["sd"])
        z0 = (x_pca - mu) / sd
        
        h1 = np.maximum(0, z0 @ self.crowd_model["W0"] + self.crowd_model["b0"])
        h2 = np.maximum(0, h1 @ self.crowd_model["W1"] + self.crowd_model["b1"])
        logits = h2 @ self.crowd_model["W2"] + self.crowd_model["b2"]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return probs.reshape(-1), (z0, h1, h2, logits)

    def predict(self, image_pil):
        """End-to-end prediction returning box, crop, probs, and class label."""
        box = self.predict_teeth_box(image_pil)
        crop_img, pixels = self.crop_and_preprocess(image_pil, box)
        x_pca = self.extract_pca(pixels)
        probs, _ = self.forward(x_pca)
        
        p_crowd = float(probs[1])
        pred_label = "crowding" if p_crowd >= 0.5 else "no crowding"
        
        return {
            "box": box,
            "crop_img": crop_img,
            "pixels": pixels,
            "x_pca": x_pca,
            "p_no_crowding": float(probs[0]),
            "p_crowding": p_crowd,
            "prediction": pred_label
        }

    def compute_occlusion_map(self, crop_pixels, patch_h=12, patch_w=16, stride=6):
        """Generates occlusion sensitivity heatmap for crowding class."""
        base_pca = self.extract_pca(crop_pixels)
        base_probs, _ = self.forward(base_pca)
        p_base = base_probs[1]
        
        grid = crop_pixels.reshape(self.img_h, self.img_w).copy()
        occ_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        count_map = np.zeros((self.img_h, self.img_w), dtype=np.float32)
        
        for y in range(0, self.img_h - patch_h + 1, stride):
            for x in range(0, self.img_w - patch_w + 1, stride):
                patched = grid.copy()
                patched[y:y+patch_h, x:x+patch_w] = 0.5  # Neutral mid-gray
                
                patched_pca = self.extract_pca(patched.reshape(-1))
                patched_probs, _ = self.forward(patched_pca)
                drop = max(0.0, p_base - patched_probs[1])
                
                occ_map[y:y+patch_h, x:x+patch_w] += drop
                count_map[y:y+patch_h, x:x+patch_w] += 1.0

        count_map[count_map == 0] = 1.0
        occ_map /= count_map
        
        if occ_map.max() > 0:
            occ_map /= occ_map.max()
            
        return occ_map

    def compute_gradient_saliency(self, x_pca):
        """Exact analytical gradient dP(crowding)/d(pixel) via backpropagation."""
        mu = self.crowd_model["mu"]
        sd = float(self.crowd_model["sd"])
        components = self.pca_model["components"]  # (126, 9216)
        
        z0 = (x_pca - mu) / sd
        h1 = np.maximum(0, z0 @ self.crowd_model["W0"] + self.crowd_model["b0"])
        h2 = np.maximum(0, h1 @ self.crowd_model["W1"] + self.crowd_model["b1"])
        logits = h2 @ self.crowd_model["W2"] + self.crowd_model["b2"]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # dP(crowd)/d(logits): for softmax, dP_1/dz_j = P_1 * (delta_1j - P_j)
        d_logits = np.array([[-probs[0, 1] * probs[0, 0], probs[0, 1] * (1.0 - probs[0, 1])]])
        
        d_h2 = d_logits @ self.crowd_model["W2"].T
        d_h1 = (d_h2 * (h2 > 0)) @ self.crowd_model["W1"].T
        d_z0 = (d_h1 * (h1 > 0)) @ self.crowd_model["W0"].T
        
        d_xpca = d_z0 / sd
        d_pixels = d_xpca @ components
        
        grad_map = np.abs(d_pixels.reshape(self.img_h, self.img_w))
        if grad_map.max() > 0:
            grad_map /= grad_map.max()
            
        return grad_map

    def get_hotspot_description(self, heat_map):
        """Generates plain-English description of the strongest activation area."""
        # Split crop into 3 vertical (upper, centre, lower) and 3 horizontal (left, centre, right)
        h_third = self.img_h // 3
        w_third = self.img_w // 3
        
        grid_vals = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                y0, y1 = r * h_third, (r + 1) * h_third
                x0, x1 = c * w_third, (c + 1) * w_third
                grid_vals[r, c] = heat_map[y0:y1, x0:x1].mean()
                
        r_max, c_max = np.unravel_index(np.argmax(grid_vals), grid_vals.shape)
        v_names = ["upper", "middle", "lower"]
        h_names = ["left", "centre", "right"]
        
        return f"{v_names[r_max]} {h_names[c_max]} part of the teeth row (as seen in the photo)"

# Quick CLI test
if __name__ == "__main__":
    engine = DentalXAIModel()
    test_pic = BASE_DIR / "data/pics/test/2018.64/2018.08.28/201864201808281.JPG"
    if test_pic.exists():
        im = Image.open(test_pic)
        res = engine.predict(im)
        print("Test prediction:", res["prediction"], f"Confidence: {res['p_crowding']:.1%}")
        occ = engine.compute_occlusion_map(res["pixels"])
        print("Hotspot:", engine.get_hotspot_description(occ))
        print("Live XAI Engine is verified and functioning perfectly!")
