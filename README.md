# Hybrid Toxic Content Detection Using BERT-Based Classification, Span Prediction, and SHAP

## Overview
This repository contains the complete implementation for the Bachelor's thesis project **"Hybrid Toxic Content Detection Using BERT-Based Classification, Span Prediction, and SHAP"**.  
The project integrates **comment-level toxicity classification**, **token-level toxic span detection**, and **post-hoc interpretability** using SHAP.  
It is based on two publicly available datasets:
- **Civil Comments** (Borkan et al., 2019) for binary comment-level toxicity classification.
- **Toxic Spans Dataset** (Pavlopoulos et al., 2022) for token-level toxic span detection.

The implementation includes:
1. **Data exploration & preprocessing**  
2. **BERT-based comment-level toxicity classification**  
3. **BERT-based token-level toxic span detection**  
4. **SHAP-based token attribution and human-span agreement evaluation**  
5. **Model comparison and visualization**

> ⚠ **Note**: The datasets and model weight files are **not** included in this repository due to size and license restrictions. Instructions to obtain them are provided below.

---

## Project Structure

- notebooks/
  - 01_explore_toxic_spans.ipynb
  - 02_prepare_civil_comments.ipynb
  - 03_train_bert_civil_comments.ipynb
  - 04_train_toxic_spans.ipynb
  - 05_explain_with_shap.ipynb
  - 06_visualize_shap_explanations.ipynb
  - 07_compare_models.ipynb

- outputs/
  - civil_comments/
  - comparison/
  - figures/
  - logs/
  - shap/
    - figures/

- .gitignore
- README.md
- requirements.txt
---

## Installation

### 1) Clone and enter the repo
```bash
git clone https://github.com/buroff1/hybrid-toxicity-detection-bert-shap.git
cd hybrid-toxicity-detection-bert-shap
```

### 2) (Recommended) Create a virtual env
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Setup

### Toxic Spans (manual file)
The notebooks expect a single CSV named `toxic_spans.csv` at:
```
data/toxic_spans.csv
```
Get it from the authors’ repo (ACL 2022 release): https://github.com/ipavlopoulos/toxic_spans/tree/master/ACL2022 
Then place/rename it exactly as `data/toxic_spans.csv`.

> Used by notebooks:
> - `01_explore_toxic_spans.ipynb`
> - `04_train_toxic_spans.ipynb`
> - `05_explain_with_shap.ipynb`
> - `06_visualize_shap_explanations.ipynb`
> - `07_compare_models.ipynb`

### Civil Comments (auto-downloaded via Hugging Face Datasets)
You **do not** need to download this manually. Notebook **`02_prepare_civil_comments.ipynb`** loads it programmatically with:
```python
from datasets import load_dataset
dataset = load_dataset("civil_comments")  # Hugging Face Datasets
```
That notebook will:
- binarize labels at `toxicity >= 0.5`,
- perform a stratified train/val/test split,
- and save CSVs to:
```
outputs/civil_comments/train.csv
outputs/civil_comments/val.csv
outputs/civil_comments/test.csv
```

> Make sure your environment has internet access for the first run so Hugging Face Datasets can fetch `civil_comments`.

---

## Quick Start

1) Launch Jupyter and open the notebooks:
```bash
jupyter lab
# or
jupyter notebook
```

2) Run in this order:
- `01_explore_toxic_spans.ipynb` (needs `data/toxic_spans.csv`)
- `02_prepare_civil_comments.ipynb` (auto-downloads Civil Comments; writes processed CSVs to `outputs/civil_comments/`)
- `03_train_bert_civil_comments.ipynb` (trains the comment classifier; saves weights under `outputs/model/`)
- `04_train_toxic_spans.ipynb` (trains the token-level span model; saves weights under `outputs/model/`)
- `05_explain_with_shap.ipynb` (computes SHAP and SHA metrics)
- `06_visualize_shap_explanations.ipynb` (renders explanation HTML/plots)
- `07_compare_models.ipynb` (comparison figures & CSVs)

