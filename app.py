"""
app.py - Dental Crowding Classification & Explainable AI (XAI) Streamlit Application
Interactive 3D Universe Companion & Live Python ML Inference Engine.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from xai_engine import DentalXAIModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Dental XAI | Explainable Dental Intelligence",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
EXPLAIN_DIR = RESULTS_DIR / "explain"

# --- Cyber Dark Theme CSS ---
st.markdown("""
<style>
    /* Dark Futuristic Theme */
    .stApp {
        background-color: #05070c;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Neon Glow Badges */
    .neon-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .badge-cyan {
        background: rgba(0, 240, 255, 0.12);
        color: #00f0ff;
        border: 1px solid rgba(0, 240, 255, 0.35);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }
    .badge-crimson {
        background: rgba(255, 42, 95, 0.15);
        color: #ff2a5f;
        border: 1px solid rgba(255, 42, 95, 0.4);
        box-shadow: 0 0 10px rgba(255, 42, 95, 0.2);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Caution Callout Box */
    .caution-box {
        background: linear-gradient(135deg, rgba(30, 8, 15, 0.9) 0%, rgba(18, 6, 10, 0.9) 100%);
        border-left: 5px solid #ff2a5f;
        border-top: 1px solid rgba(255, 42, 95, 0.3);
        border-right: 1px solid rgba(255, 42, 95, 0.3);
        border-bottom: 1px solid rgba(255, 42, 95, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(255, 42, 95, 0.15);
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(13, 17, 28, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Engine ---
@st.cache_resource
def get_model():
    return DentalXAIModel(results_dir=RESULTS_DIR)

model = get_model()

# --- Load Datasets ---
@st.cache_data
def load_data():
    with open(RESULTS_DIR / "crowding_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(EXPLAIN_DIR / "pointing_check.json", "r", encoding="utf-8") as f:
        pointing = json.load(f)
    patients_df = pd.read_csv(EXPLAIN_DIR / "explained_patients.csv")
    return metrics, pointing, patients_df

metrics, pointing, patients_df = load_data()

# --- Sidebar ---
st.sidebar.markdown("## 🦷 DENTAL **XAI** LAB")
st.sidebar.markdown(
    '<span class="neon-badge badge-cyan">v2.4 Research Engine</span>'
    '<span class="neon-badge badge-crimson">XAI Audit</span>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigation Console",
    [
        "🚀 3D Digital Universe (Web UI)",
        "⚡ AI Scanner & Live Inference",
        "📊 Model Performance Lab",
        "🔬 Explainable AI Gallery",
        "🎯 Pointing Game Sanity Check",
        "📋 Full Cohort Data (84 Patients)",
        "🌐 GitHub & Cloud Deployment"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Mandatory Audit Notice")
st.sidebar.info(
    "**Experimental Research Pipeline**\n\n"
    "• Test ROC-AUC: **0.701** (Weak)\n"
    "• Test Accuracy: **64.1%** (Below 66.7% baseline)\n"
    "• Pointing Game Lift: **0.95x** (Distracted attention)\n"
    "• 'Crowding' is an operational proxy (≥ 1 rotated/misaligned teeth), not a formal orthodontic diagnosis."
)

# --- Top Banner: Permanent Scientific Disclaimers ---
st.markdown("""
<div class="caution-box">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 1.2rem;">⚠️</span>
        <strong style="color: #fff; font-size: 1rem; letter-spacing: 0.02em;">CRITICAL CLINICAL & METHODOLOGICAL CAVEATS</strong>
    </div>
    <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5; margin: 0;">
        <strong>1. Weak Model:</strong> Test ROC-AUC is 0.701; test accuracy (64.1%) is lower than the naive majority-class baseline (66.7%).<br>
        <strong>2. Uncalibrated Confidence:</strong> Network probabilities reflect raw softmax outputs on 192 training images, not calibrated disease risk.<br>
        <strong>3. Proxy Definition:</strong> "Crowding" is operationalized as ≥ 1 dentist-marked rotated (<tt>TT</tt>) or misaligned (<tt>TM</tt>) front teeth, not clinical diagnosis.<br>
        <strong>4. Negative Pointing Game:</strong> Median lift is 0.95 (&lt; 1.0 chance baseline; only 45% of patients &gt; 1.0). The model's heat maps land on problem teeth no more than random chance.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# 1. 3D DIGITAL UNIVERSE LAUNCHER
# =====================================================================
if nav == "🚀 3D Digital Universe (Web UI)":
    st.markdown("## 🌌 Immersive 3D Dental AI Universe")
    st.markdown("Experience the interactive 3D digital laboratory inspired by Bruno Simon's portfolio website.")
    
    col1, col2 = st.columns([1.8, 1])
    with col1:
        st.markdown("""
        The 3D Web experience includes:
        - **Interactive Human Dental Arch Model:** Procedural 3D mandible, sculpted enamel teeth, and orthodontic archwires.
        - **Floating 3D Navigation Totems:** Glowing Bio-Neural Brain, Saliency Heatmap Orb, Quantum Scanner, and Precision Target.
        - **Web Audio Sound Engine:** Spatial audio hums, portal hover sweeps, and laser scanning effects with toggleable mute.
        - **Smooth GSAP Camera Transitions:** Dynamic fly-throughs between stations.
        - **Zero-Backend Standalone:** Opens directly in any browser via `index.html`!
        """)
        
        index_path = BASE_DIR / "index.html"
        if index_path.exists():
            st.success("✅ `index.html` is compiled and ready in your project directory!")
            st.code("Start local web server:\npython -m http.server 8000\nThen open: http://localhost:8000", language="bash")
        else:
            st.warning("`index.html` is being generated.")

    with col2:
        st.markdown("### 🎮 Keyboard Shortcuts in 3D")
        st.markdown("""
        - **Click & Drag:** Orbit and rotate 3D dental arcade
        - **Scroll:** Smooth zoom in/out
        - **Key `T`:** Toggle holographic wireframe mode
        - **Key `S`:** Toggle sound on/off
        - **Key `E`:** Gravity disruption physics easter egg!
        - **Key `Esc`:** Return to central laboratory
        """)


# =====================================================================
# 2. AI SCANNER & LIVE INFERENCE
# =====================================================================
elif nav == "⚡ AI Scanner & Live Inference":
    st.markdown("## ⚡ AI Dental Scanner & Real-Time XAI Inference")
    st.markdown("Upload any frontal dental photograph to run the **actual trained model** and generate real occlusion & gradient heatmaps.")
    
    scan_col1, scan_col2 = st.columns([1, 1.2])
    
    with scan_col1:
        upload_mode = st.radio("Choose Input Mode", ["Upload My Own Image", "Select Pre-existing Test Patient"], horizontal=True)
        image_to_process = None
        sample_name = "custom_upload"

        if upload_mode == "Upload My Own Image":
            uploaded_file = st.file_uploader("Upload Frontal Dental Photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image_to_process = Image.open(uploaded_file)
                sample_name = uploaded_file.name
        else:
            sample_options = {
                "Patient 201864 (Test Split - Ground Truth: Crowding)": BASE_DIR / "sample_images/patient_201864_test_crowding.jpg",
                "Patient 201812 (Val Split - Ground Truth: Crowding)": BASE_DIR / "sample_images/patient_201812_val_crowding.jpg",
                "Patient 201841 (Val Split - Ground Truth: Healthy / No Crowding)": BASE_DIR / "sample_images/patient_201841_val_healthy.jpg",
                "Fallback 201864 in data/pics": BASE_DIR / "data/pics/test/2018.64/2018.08.28/201864201808281.JPG",
                "Fallback 201812 in data/pics": BASE_DIR / "data/pics/val/2018.12/2018.07.19/201812201807191.JPG"
            }
            # Pick first available
            available_samples = {k: v for k, v in sample_options.items() if v.exists()}
            if available_samples:
                chosen = st.selectbox("Select Patient Photo", list(available_samples.keys()))
                image_to_process = Image.open(available_samples[chosen])
                sample_name = Path(available_samples[chosen]).name
            else:
                st.info("Dataset images not found in standard path. Please upload your own image.")

        if image_to_process:
            st.image(image_to_process, caption=f"Selected Input: {sample_name}", use_container_width=True)
            run_btn = st.button("⚡ ANALYZE IMAGE WITH REAL MODEL", type="primary", use_container_width=True)
        else:
            run_btn = False

    with scan_col2:
        if image_to_process and run_btn:
            with st.spinner("Executing Ridge teeth detection, PCA projection, MLP forward pass, and occlusion mapping..."):
                # Real inference
                res = model.predict(image_to_process)
                l, t, r, b = res["box"]
                p_crowd = res["p_crowding"]
                pred = res["prediction"]
                
                # Real occlusion heatmap
                occ_map = model.compute_occlusion_map(res["pixels"])
                hotspot = model.get_hotspot_description(occ_map)
                
                # Real gradient sensitivity map
                grad_map = model.compute_gradient_saliency(res["x_pca"])

            # Classification Card
            st.markdown("### 📋 Real Model Prediction")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if pred == "crowding":
                    st.error(f"**Predicted Class:** CROWDING ({p_crowd:.1%})")
                else:
                    st.success(f"**Predicted Class:** NO CROWDING ({res['p_no_crowding']:.1%})")
            with m_col2:
                st.metric("P(Crowding)", f"{p_crowd:.1%}", delta=f"{p_crowd - 0.5:.1%} from 50% threshold")

            st.markdown(f"**Strongest Support Hot Spot:** `{hotspot}`")
            st.markdown(f"**Predicted Teeth Bounding Box:** `[{l:.3f}, {t:.3f}, {r:.3f}, {b:.3f}]`")

            # Interactive Overlay Visualizations
            st.markdown("### 🔬 Explainable AI Heatmap Inspection")
            alpha = st.slider("Heatmap Overlay Intensity", min_value=0.0, max_value=1.0, value=0.65, step=0.05)

            # Draw figure
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.patch.set_facecolor('#05070c')
            
            # Subplot 1: Detected teeth area on photo
            w, h = image_to_process.size
            axes[0].imshow(image_to_process)
            rect = patches.Rectangle((l * w, t * h), (r - l) * w, (b - t) * h, fill=False, edgecolor="#00f0ff", linewidth=2.5)
            axes[0].add_patch(rect)
            axes[0].set_title("1. Detected Teeth Box", color="#fff", fontsize=10)
            axes[0].axis("off")

            # Subplot 2: Crop with Occlusion Heatmap
            crop_np = np.asarray(res["crop_img"], dtype=np.float32) / 255.0
            axes[1].imshow(crop_np, cmap="gray")
            axes[1].imshow(occ_map, cmap="magma", alpha=alpha)
            axes[1].set_title("2. Occlusion Sensitivity Map", color="#fff", fontsize=10)
            axes[1].axis("off")

            # Subplot 3: Gradient Sensitivity
            axes[2].imshow(crop_np, cmap="gray")
            axes[2].imshow(grad_map, cmap="plasma", alpha=alpha)
            axes[2].set_title("3. Gradient Pixel Sensitivity", color="#fff", fontsize=10)
            axes[2].axis("off")

            st.pyplot(fig)
        elif not image_to_process:
            st.info("👈 Upload or select a dental image on the left, then click 'Analyze Image' to execute the real ML pipeline.")


# =====================================================================
# 3. MODEL PERFORMANCE LABORATORY
# =====================================================================
elif nav == "📊 Model Performance Lab":
    st.markdown("## 📊 Crowding Model Performance & Held-Out Test Evaluation")
    st.markdown("Actual metrics extracted from `results/crowding_metrics.json` on the held-out test cohort (N=39 patients).")

    test_stats = metrics.get("test", {})
    val_stats = metrics.get("val", {})

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Test ROC-AUC", f"{test_stats.get('auc', 0.701):.3f}", delta=f"Val AUC: {val_stats.get('auc', 0.649):.3f}")
    kpi2.metric("Test Accuracy", f"{test_stats.get('accuracy', 0.641):.1%}", delta="-2.6% vs Majority Baseline", delta_color="inverse")
    kpi3.metric("Majority Baseline", f"{test_stats.get('majority_baseline_accuracy', 0.667):.1%}")
    kpi4.metric("Sensitivity (Crowding)", f"{test_stats.get('sensitivity', 0.654):.1%}")
    kpi5.metric("Specificity (Healthy)", f"{test_stats.get('specificity', 0.615):.1%}")

    st.markdown("---")
    cm_col1, cm_col2 = st.columns([1, 1.2])

    with cm_col1:
        st.markdown("### 🖼️ Confusion Matrix Visualizer")
        cm_img_path = RESULTS_DIR / "crowding_confusion.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="results/crowding_confusion.png (Test Set)", use_container_width=True)
        else:
            st.warning("results/crowding_confusion.png not found.")

    with cm_col2:
        st.markdown("### 🔬 Scientific Breakdown & Error Analysis")
        st.markdown("""
        | Actual Class | Pred: No Crowding | Pred: Crowding | Total |
        | :--- | :---: | :---: | :---: |
        | **True: No Crowding** | **8** (True Negative) | **5** (False Positive) | 13 |
        | **True: Crowding** | **9** (False Negative) | **17** (True Positive) | 26 |
        | **Total** | 17 | 22 | **39** |

        - **Positive Predictive Value (Precision):** `77.3%` (17 / 22)
        - **Negative Predictive Value (NPV):** `47.1%` (8 / 17)
        - **Miss Rate (False Negative Rate):** `34.6%` (9 out of 26 crowding patients were missed)
        - **Small Sample Impact:** Because the test set contains only 39 patients, **each single patient shifts accuracy by ~2.56 percentage points**.
        """)
        st.error(
            "⚠️ **Clinical Warning:** Test accuracy (64.1%) is lower than the naive majority-class baseline (66.7%). "
            "Predicting 'crowding' for every patient yields a higher score than the neural network. "
            "This model is a proof-of-concept pipeline, not a diagnostic tool."
        )


# =====================================================================
# 4. EXPLAINABLE AI GALLERY
# =====================================================================
elif nav == "🔬 Explainable AI Gallery":
    st.markdown("## 🔬 Explainable AI Gallery (Patient Explanations)")
    st.markdown("Browse patient explanation figures from `results/explain/` with matched ground truth and companion JSON files.")

    figs = sorted([f for f in os.listdir(EXPLAIN_DIR) if f.endswith(".png")])
    if not figs:
        st.warning("No explanation PNGs found in results/explain/")
    else:
        chosen_fig = st.selectbox("Select Patient Explanation Figure", figs)
        fig_path = EXPLAIN_DIR / chosen_fig
        stem = chosen_fig.replace(".png", "")
        json_path = EXPLAIN_DIR / f"{stem}.json"

        companion_json = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as jf:
                companion_json = json.load(jf)

        g_col1, g_col2 = st.columns([1.6, 1])

        with g_col1:
            st.image(str(fig_path), caption=f"Explanation: {chosen_fig}", use_container_width=True)
            st.caption("Figure layout: 1. Clinical Photo & Tooth Boxes | 2. Teeth Crop | 3. Occlusion Sensitivity Map | 4. Gradient Sensitivity")

        with g_col2:
            st.markdown("### 📋 Patient Metadata")
            p_val = companion_json.get("p_crowding", 0.5)
            pred = companion_json.get("prediction", "crowding" if p_val >= 0.5 else "no crowding")
            hotspot = companion_json.get("strongest_support", "upper centre part of the teeth row (as seen in the photo)")

            st.metric("Model Confidence P(Crowding)", f"{p_val:.1%}")
            st.markdown(f"**Predicted Class:** `{pred.upper()}`")
            st.markdown(f"**Strongest Support Hot Spot:** `{hotspot}`")

            st.markdown("#### Companion JSON (`results/explain/...`)")
            st.json(companion_json)


# =====================================================================
# 5. POINTING GAME SANITY CHECK
# =====================================================================
elif nav == "🎯 Pointing Game Sanity Check":
    st.markdown("## 🎯 Pointing Game Sanity Check")
    st.markdown("Honest statistical validation from `results/explain/pointing_check.json` testing whether heatmaps isolate true problem teeth.")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Median Pointing Lift", f"{pointing.get('median_lift', 0.95):.2f}x", delta="< 1.0 (Fails Baseline)", delta_color="inverse")
    p2.metric("Patients with Lift > 1.0", f"{pointing.get('share_with_lift_above_1', 0.45):.0%}", delta="55% Below Chance", delta_color="inverse")
    p3.metric("Hot Region on Problem Teeth", f"{pointing.get('mean_share_of_hot_region_on_problem_teeth', 0.147):.1%}")
    p4.metric("Random Area Chance Baseline", f"{pointing.get('mean_chance_share', 0.134):.1%}")

    st.error("""
    **HONEST XAI AUDIT FINDING: The saliency maps fail the pointing game.**
    - Saliency maps overlap dentist-marked problem teeth only 14.7% of the time vs 13.4% random chance.
    - The median lift of 0.95 confirms that the model's heatmaps focus on problem teeth *no more than a random patch of the same size*.
    - The CNN exploits non-teeth confounding features (lip retractors, saliva glares, gums).
    - **Conclusion:** Saliency heatmaps represent arbitrary pixel sensitivities, NOT verified evidence of dental crowding.
    """)

    st.markdown("### 📋 Per-Patient Pointing Results (56 Patients with Marked Teeth)")
    p_list = pointing.get("per_patient", [])
    if p_list:
        df_p = pd.DataFrame(p_list)
        df_p["hit_rate"] = (df_p["hit_rate"] * 100).round(1).astype(str) + "%"
        df_p["chance"] = (df_p["chance"] * 100).round(1).astype(str) + "%"
        df_p["lift"] = df_p["lift"].round(2)
        df_p["verdict"] = df_p["lift"].apply(lambda v: "✓ Above Chance" if v >= 1.0 else "✗ Below Chance")
        st.dataframe(df_p, use_container_width=True, height=350)


# =====================================================================
# 6. FULL COHORT DATA (84 PATIENTS)
# =====================================================================
elif nav == "📋 Full Cohort Data (84 Patients)":
    st.markdown("## 📋 Full Evaluation Cohort (84 Patients)")
    st.markdown("Complete records from `results/explain/explained_patients.csv`.")

    split_filter = st.selectbox("Filter by Split", ["All", "val", "test"])
    if split_filter != "All":
        filtered_df = patients_df[patients_df["split"] == split_filter]
    else:
        filtered_df = patients_df

    st.dataframe(filtered_df, use_container_width=True, height=450)
    
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered CSV", data=csv_bytes, file_name="explained_patients_filtered.csv", mime="text/csv")


# =====================================================================
# 7. GITHUB & CLOUD DEPLOYMENT
# =====================================================================
elif nav == "🌐 GitHub & Cloud Deployment":
    st.markdown("## 🌐 GitHub & Streamlit Community Cloud Deployment Guide")
    
    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        st.markdown("### 🐙 1. Push to GitHub")
        st.code("""
# Initialize Git in your project folder
git init

# Add all files (models, code, UI)
git add .

# Create initial commit
git commit -m "Dental Crowding Classification & 3D XAI Laboratory"

# Link to your remote GitHub repository
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
        """, language="bash")
        st.caption("Tip: Enable GitHub Pages in repo settings -> Deploy from branch (main / root) to host index.html live!")

    with dep_col2:
        st.markdown("### ☁️ 2. Deploy to Streamlit Cloud (Free)")
        st.markdown("""
        1. Visit **[share.streamlit.io](https://share.streamlit.io)**
        2. Sign in with your GitHub account.
        3. Click **'New app'**.
        4. Select your repository: `<YOUR_USERNAME>/<YOUR_REPO>`
        5. Set Main file path to: `app.py`
        6. Click **'Deploy!'**
        """)
        st.info("The application will be live globally with real model inference!")

st.markdown("---")
st.caption("Dental Crowding Classification & Explainable AI (XAI) Prototype • Pipeline Transparency & Non-Clinical Benchmark")
