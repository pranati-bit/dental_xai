# SEE BEYOND THE SMILE: 3D Dental AI & Explainability Laboratory

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-00f0ff.svg)](https://threejs.org/)
[![Status](https://img.shields.io/badge/Status-XAI%20Research%20Audit-ff2a5f.svg)]()

> **"SEE BEYOND THE SMILE."**  
> *Artificial Intelligence. Dental Intelligence. Explainable Decisions.*

---

## 🌌 Overview & Creative Direction

Inspired by the playful 3D exploration and unconventional navigation of **Bruno Simon's portfolio**, this project transforms a conventional dental machine-learning benchmark into an **interactive 3D digital universe**.

Rather than a static dashboard, users explore a futuristic **dark-mode 3D dental laboratory** featuring:
- **Procedural 3D Human Dental Arcade:** Sculpted enamel teeth along an orthodontic catenary arch, reflective metallic archwires, brackets, and interactive lighting.
- **Floating 3D Navigation Totems:** Glowing Bio-Neural Brain, Saliency Heatmap Orb, Quantum Laser Scanner, and Precision Target portals.
- **Web Audio Sound Engine:** Interactive spatial audio, hover frequencies, laser scan sweeps, and toggleable sound effects.
- **GSAP Cinematic Camera Transitions:** Smooth dynamic fly-throughs between laboratory stations.
- **Real Python Inference Engine:** Live prediction and XAI occlusion/gradient map generation on user-uploaded dental photographs.

---

## ⚠️ Mandatory Scientific Disclaimers & Model Caveats

This project adheres to strict clinical and scientific honesty. The following caveats are permanently surfaced across both the 3D Web UI and the Streamlit app:

1. **Weak Model Performance:**
   - **Test ROC-AUC:** `0.701` (Validation AUC: `0.649`).
   - **Test Accuracy:** `64.1%` (25/39 test patients), which is **below the naive majority-class baseline** (`66.7%`). A dummy classifier predicting "crowding" for all patients achieves higher accuracy.
2. **Uncalibrated Confidence:**
   - Softmax probabilities cluster at extremes due to uncalibrated network weights fit on a small training cohort ($N=192$). These reflect internal logit scales, **not calibrated patient disease risk**.
3. **Operational Proxy Definition:**
   - "Crowding" is operationalized as $\ge 1$ dentist-annotated rotated (`TT`) or misaligned (`TM`) front teeth. It does **not** replace clinical diagnostic indexes (e.g., Little's Irregularity Index, Bolton ratio, or arch space discrepancy).
4. **Negative Pointing Game Finding:**
   - **Median Lift:** `0.95x` (below the 1.0 chance baseline).
   - **Share with Lift > 1.0:** `45.0%` (55% of patients fail the sanity check).
   - The model's heatmaps land on problem teeth *no more than random chance*. The neural network exploits non-teeth confounding features such as lip retractors, saliva reflections, and gum lines. Heatmaps must be interpreted as "what arbitrary pixels stimulated the weights", **never as clinical proof of crowding**.

---

## 🏗️ Architecture & Key Components

```
braces_proj_code/
├── index.html                  # Standalone 3D WebGL experience (Three.js + GSAP + Audio)
├── app.py                      # Live Streamlit Python application & inference station
├── xai_engine.py               # Real model inference & XAI heatmap engine
├── requirements.txt            # Python dependencies (Streamlit, NumPy, Pandas, Pillow, etc.)
├── results/
│   ├── crowding_metrics.json   # Held-out evaluation metrics
│   ├── crowding_confusion.png  # Actual test set confusion matrix image
│   ├── box_model.npz           # Ridge regression teeth bounding box detector
│   ├── pca_params.npz          # PCA projection matrix (9216 -> 126 components)
│   ├── crowding_model.npz      # Trained MLP weights (126 -> 16 -> 8 -> 2)
│   └── explain/
│       ├── pointing_check.json # Pointing game evaluation across 56 patients
│       ├── explained_patients.csv # Cohort record of all 84 patients
│       └── *.png / *.json      # Explanation figures & companion metadata
```

---

## 🚀 Quickstart: Running Locally

### Option 1: Launch the 3D Web UI (`index.html`)
The 3D universe is completely self-contained with zero backend dependencies. You can open `index.html` directly in any modern browser:
```bash
# Optional: run a local static web server
python -m http.server 8000
# Then visit: http://localhost:8000
```

### Option 2: Run the Streamlit Application (`app.py`)
To run the live Python machine-learning inference station:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit
streamlit run app.py
```

---

## 🐙 How to Upload to GitHub

Follow these steps in your terminal inside the project directory:

```bash
# 1. Initialize Git repository
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "Initialize 3D Dental AI & Explainability Laboratory"

# 4. Set main branch
git branch -M main

# 5. Connect to your GitHub repository
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git

# 6. Push to GitHub
git push -u origin main
```

> **Tip (GitHub Pages):** Go to your GitHub repository **Settings** → **Pages** → Source: **Deploy from a branch** → Branch: **main** / **root** → Click **Save**. Your 3D Web UI will be hosted live on the web for free!

---

## ☁️ How to Deploy on Streamlit Community Cloud (Free)

1. Push your repository to GitHub (following the steps above).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New app"**.
4. Select your repository: `<YOUR_USERNAME>/<YOUR_REPO_NAME>`.
5. Set the Main file path to: `app.py`.
6. Click **"Deploy!"**.

Your live AI scanner and interactive dashboard will be accessible worldwide!

---

## 📜 Ethical & Educational Disclosure
This system is an academic research and explainability audit prototype. It is not approved by medical regulatory bodies and must never be utilized for direct clinical diagnosis or orthodontic treatment planning.
