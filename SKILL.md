---
name: research-pipeline
description: >
  Use this skill whenever the user wants to convert Jupyter notebook experiments
  into a clean, publication-ready GitHub research repository. Triggers include:
  "push my experiments to GitHub", "make a research repo", "clean up my notebook
  for publication", "build a pipeline from my experiments", "make a README for my
  paper", "organize my ML research code", "prepare code for paper submission",
  or any mention of turning notebooks into a structured project with README,
  requirements, and proper folder layout. This skill behaves like a professional
  ML researcher and research engineer who knows how to structure repos for
  reproducibility. Always use this skill when the user mentions a research
  notebook + GitHub + paper submission in the same context.
---

# Research Pipeline Skill

You are acting as a **professional ML researcher and research engineer**. Your job is to transform the user's experimental Jupyter notebooks into a clean, reproducible, publication-quality GitHub repository — the kind that gets cited and starred.

---

## Context: What This Skill Builds

The user has:
- One or more **Jupyter notebooks** containing experiments
- A **proposed method** (e.g., DenseNet121 + 4-qubit 2-layer quantum circuit = Hybrid QNN)
- **Ablation / baseline experiments** comparing the proposed method against others
- A research paper in progress — the GitHub repo link will appear in the paper

You will produce a repo with this structure and all supporting files.

---

## Step 0: Understand the Research Before Touching Any Files

Before writing a single line, ask or infer:

1. **What is the proposed method?** (e.g., DenseNet121 + CBAM + Quantum Circuit, 4 qubits, 2 layers)
2. **What are the ablation/baseline models?** (e.g., DenseNet121-only, ResNet50, VGG16, classical CNN)
3. **What dataset is used?** (e.g., MSLD v2.0, monkeypox, ISIC)
4. **What metrics are reported?** (accuracy, AUC, F1, sensitivity, specificity)
5. **What framework?** (PyTorch / TensorFlow / PennyLane)
6. **What is the paper title / short name?** (used for repo name and README header)

If the user pastes a notebook, extract this info from it. Do not ask redundant questions.

---

## Step 1: Define the Repository Structure

Generate the following folder layout based on what is in the notebooks:

```
repo-name/
├── README.md                  ← Main entry point (see Step 2)
├── requirements.txt           ← Pinned dependencies (see Step 3)
├── environment.yml            ← Conda env (optional but recommended)
├── setup.py                   ← Optional, if installable package
│
├── data/
│   ├── README.md              ← Describe dataset, download link, splits
│   └── splits/                ← train/val/test split CSVs or JSONs
│
├── models/
│   ├── __init__.py
│   ├── proposed.py            ← Proposed hybrid/novel model
│   ├── baselines.py           ← All comparison models
│   └── blocks.py              ← Shared building blocks (CBAM, QuantumLayer, etc.)
│
├── train.py                   ← Main training script (CLI args)
├── evaluate.py                ← Evaluation / inference script
├── predict.py                 ← Single-image prediction (optional)
│
├── configs/
│   ├── proposed.yaml          ← Hyperparams for proposed method
│   └── baselines/
│       ├── densenet121.yaml
│       └── resnet50.yaml
│
├── notebooks/
│   ├── 01_EDA.ipynb           ← Exploratory data analysis
│   ├── 02_proposed_method.ipynb
│   ├── 03_ablation_study.ipynb
│   └── 04_results_visualization.ipynb
│
├── results/
│   ├── figures/               ← ROC curves, confusion matrices, Grad-CAM
│   └── tables/                ← CSV exports of metric tables
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── losses.py
│   ├── augmentation.py
│   └── visualization.py       ← Grad-CAM, SHAP, LIME helpers
│
├── tests/
│   └── test_models.py         ← Basic forward-pass smoke tests
│
├── .gitignore
├── LICENSE                    ← MIT recommended for research
└── CITATION.cff               ← For GitHub "Cite this repository"
```

Adapt the structure to what's actually in the notebooks. Don't invent folders for things that don't exist.

---

## Step 2: Write the README.md

The README is the face of the paper's code. Follow this template exactly:

```markdown
# [Paper Title]

> [One-sentence tagline: what problem, what method, what result]

[![Paper](https://img.shields.io/badge/paper-arxiv-red)](LINK)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

---

## 📌 Abstract

[2–3 sentence summary of the paper, written in plain English]

---

## 🏗️ Architecture

[Insert architecture diagram if available, or describe it in text]

**Proposed Method**: [e.g., DenseNet121 backbone → CBAM attention → Quantum Feature Extractor (4 qubits, 2 layers) → Classifier]

---

## 📊 Results

| Model | Accuracy | AUC | F1 | Sensitivity | Specificity |
|-------|----------|-----|-----|-------------|-------------|
| **Proposed (Ours)** | **XX.XX%** | **X.XXX** | ... | ... | ... |
| DenseNet121 (Baseline) | XX.XX% | ... | ... | ... | ... |
| ResNet50 | ... | ... | ... | ... | ... |

*Tested on [Dataset Name]. See `results/` for full tables and figures.*

---

## 🚀 Quick Start

### 1. Clone & Install
\```bash
git clone https://github.com/username/repo-name
cd repo-name
pip install -r requirements.txt
\```

### 2. Prepare Data
\```bash
# Download [Dataset] from [URL]
# Place in data/ as described in data/README.md
\```

### 3. Train
\```bash
python train.py --config configs/proposed.yaml
\```

### 4. Evaluate
\```bash
python evaluate.py --checkpoint checkpoints/best.pth --config configs/proposed.yaml
\```

---

## 📁 Repository Structure

[Paste abbreviated tree here]

---

## 🔬 Experiments

All experiments are reproducible via the notebooks in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Dataset analysis and class distribution |
| `02_proposed_method.ipynb` | Full training + evaluation of proposed model |
| `03_ablation_study.ipynb` | Component-wise ablation experiments |
| `04_results_visualization.ipynb` | Figures and metric tables for paper |

---

## 📦 Requirements

See `requirements.txt`. Key dependencies:
- Python 3.8+
- PyTorch X.X / TensorFlow X.X
- PennyLane X.X (for quantum layers)
- [other key deps]

---

## 📖 Citation

If you use this code in your research, please cite:

\```bibtex
@article{[citationkey][year],
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {[Journal/Conference]},
  year    = {[Year]},
  url     = {[URL]}
}
\```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

[Funding, dataset sources, framework credits]
```

---

## Step 3: Write requirements.txt

Extract all imports from the notebook(s) and generate a pinned requirements file. Use this format:

```
# Core ML
torch==2.1.0
torchvision==0.16.0
tensorflow==2.13.0         # remove if PyTorch-only

# Quantum ML
pennylane==0.36.0
pennylane-lightning==0.36.0

# Data & Visualization
numpy==1.24.4
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
opencv-python==4.8.0.76
Pillow==10.0.0

# Explainability
shap==0.43.0
lime==0.2.0.1
grad-cam==1.4.8

# Utilities
tqdm==4.66.1
PyYAML==6.0.1
jupyter==1.0.0
ipykernel==6.25.2
```

Always include a comment block at the top:
```
# Requirements for: [Paper Title]
# Python 3.8+ required
# Install with: pip install -r requirements.txt
# For GPU support: install CUDA 11.8+ and the GPU version of torch
```

---

## Step 4: Write environment.yml (Conda)

```yaml
name: [repo-short-name]
channels:
  - defaults
  - conda-forge
  - pytorch
dependencies:
  - python=3.8
  - pip
  - pip:
    - -r requirements.txt
```

---

## Step 5: Write CITATION.cff

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "[Last]"
    given-names: "[First]"
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
title: "[Paper Title]"
version: 1.0.0
date-released: "[YYYY-MM-DD]"
url: "https://github.com/username/repo-name"
preferred-citation:
  type: article
  authors:
    - family-names: "[Last]"
      given-names: "[First]"
  title: "[Paper Title]"
  journal: "[Journal]"
  year: [year]
```

---

## Step 6: Write .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (never commit raw data)
data/raw/
data/processed/
*.h5
*.hdf5
*.npy
*.npz
*.csv

# Model checkpoints (use Git LFS or release assets)
checkpoints/
*.pth
*.pt
*.ckpt
*.pkl
*.h5

# Results (keep only final figures/tables)
results/raw/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## Step 7: Write data/README.md

```markdown
# Dataset

## [Dataset Name]

- **Source**: [URL or citation]
- **Size**: [N images, K classes]
- **Split**: [Train/Val/Test counts]
- **License**: [CC-BY, research-only, etc.]

## Setup

1. Download from [URL]
2. Extract to `data/raw/`
3. Run preprocessing: `python utils/preprocess.py`
4. Splits are saved to `data/splits/`

## Class Distribution

| Class | Train | Val | Test |
|-------|-------|-----|------|
| [Class A] | N | N | N |
| [Class B] | N | N | N |
```

---

## Step 8: Refactor Notebooks

When cleaning up notebooks:

1. **Rename** them with numbered prefixes: `01_`, `02_`, etc.
2. **Add a markdown cell at the top** of each:
   ```markdown
   # [Notebook Title]
   **Paper**: [Title]  
   **Authors**: [Names]  
   **Description**: What this notebook does in 1–2 sentences.
   **Runtime**: ~X minutes on [GPU/CPU]
   ```
3. **Add a requirements cell** at top:
   ```python
   # !pip install -r ../requirements.txt
   ```
4. **Clear all outputs before committing** (add to .gitignore or use `nbstripout`)
5. **Ensure reproducibility**: set random seeds at top of each notebook
   ```python
   import random, numpy as np, torch
   SEED = 42
   random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
   ```

---

## Step 9: Generate the Files

After gathering all information, generate the files in this order:

1. `README.md` — most visible, do this first
2. `requirements.txt`
3. `.gitignore`
4. `CITATION.cff`
5. `environment.yml`
6. `data/README.md`
7. Any model files extracted from notebooks (`models/proposed.py`, etc.)
8. `train.py` skeleton (if user wants runnable scripts, not just notebooks)

Always create real files with `create_file` tool and then call `present_files` to share them.

---

## Quality Checklist

Before finalizing, verify:

- [ ] README has results table with numbers filled in
- [ ] README has working Quick Start commands
- [ ] requirements.txt has all imports from notebooks
- [ ] .gitignore excludes checkpoints and raw data
- [ ] CITATION.cff is present with correct BibTeX
- [ ] Notebooks are numbered and have header cells
- [ ] Seed is set for reproducibility
- [ ] Data download instructions are clear
- [ ] Repo structure matches what's actually in the notebooks

---

## Reference Files

- See `references/readme_examples.md` for README examples from top ML papers
- See `references/requirements_patterns.md` for common ML stack combinations
