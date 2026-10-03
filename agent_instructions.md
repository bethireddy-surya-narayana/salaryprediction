# Agent Instructions

## Project Overview
This is a binary classification ML web application predicting adult income (>50K / <=50K)
using the UCI Adult Census Income dataset.

## Dataset
- **File:** `data/adult.csv`
- **Rows:** 48,842
- **Target column:** `income` (binary: `<=50K` = 0, `>50K` = 1)
- **Missing values:** encoded as `?` in the raw CSV — handled in `train_model.py`

## Key Files
| File | Purpose |
|---|---|
| `train_model.py` | Full ML pipeline: load → preprocess → compare models → train → save |
| `app.py` | Streamlit UI: load `models/model.pkl` → accept inputs → predict |
| `models/model.pkl` | Serialized sklearn Pipeline + metadata dict (joblib) |
| `data/adult.csv` | Raw dataset — do NOT modify |

## How the Model Artifact Works
`models/model.pkl` is a Python `dict` with keys:
- `pipeline` — sklearn `Pipeline` (preprocessor + best classifier)
- `model_name` — string name of the winning algorithm
- `features` — list of feature column names in training order
- `num_cols` / `cat_cols` — numeric vs categorical feature lists
- `target` — target column name
- `label_map` — `{0: "<=50K", 1: ">50K"}`
- `test_accuracy`, `test_f1`, `test_auc` — held-out test metrics
- `cat_unique` — dict mapping each categorical column → sorted unique values (for Streamlit dropdowns)
- `num_stats` — `describe()` stats dict for numeric columns

## Retraining
```bash
python train_model.py
```
Re-running will overwrite `models/model.pkl` with the newly trained pipeline.

## Extending
- To add a new model, add an entry to the `CANDIDATES` dict in `train_model.py`.
- To add a new feature, update `FEATURES` handling and add the corresponding input widget in `app.py`.

## Deployment Notes
- All paths are relative (`pathlib.Path(__file__).parent`).
- No API keys or secrets required.
- Compatible with Streamlit Community Cloud — just ensure `models/model.pkl` is committed.
