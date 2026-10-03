"""
train_model.py
--------------
Reads the Adult Income dataset, preprocesses it, trains and compares multiple
classification models, selects the best by F1-score, and saves the complete
sklearn Pipeline to models/model.pkl.

Run:
    python train_model.py
"""

import os
import pathlib
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "adult.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("ADULT INCOME — SALARY PREDICTION (Binary Classification)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Identify target and features
# ---------------------------------------------------------------------------
TARGET = "income"
FEATURES = [c for c in df.columns if c != TARGET]

# Replace '?' with NaN so imputers can handle them
df.replace("?", np.nan, inplace=True)

# Strip whitespace from string columns
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# Encode target: <=50K → 0, >50K → 1
df[TARGET] = (df[TARGET].str.strip() == ">50K").astype(int)

print(f"[2] Target column : '{TARGET}'")
print(f"    Classes        : 0=<=50K ({(df[TARGET]==0).sum():,}), "
      f"1=>50K ({(df[TARGET]==1).sum():,})")

# Drop 'fnlwgt' (census weight — not a predictive feature for individuals)
df.drop(columns=["fnlwgt"], inplace=True)
FEATURES = [c for c in df.columns if c != TARGET]

# ---------------------------------------------------------------------------
# 3. Split numeric / categorical columns
# ---------------------------------------------------------------------------
NUM_COLS = df[FEATURES].select_dtypes(include=["int64", "float64"]).columns.tolist()
CAT_COLS = df[FEATURES].select_dtypes(include=["object"]).columns.tolist()

print(f"\n[3] Numeric  features ({len(NUM_COLS)}): {NUM_COLS}")
print(f"    Categorical features ({len(CAT_COLS)}): {CAT_COLS}")

# ---------------------------------------------------------------------------
# 4. Build preprocessing pipeline
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUM_COLS),
    ("cat", categorical_transformer, CAT_COLS),
])

# ---------------------------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[4] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 6. Model candidates
# ---------------------------------------------------------------------------
# Lighter estimators used during CV selection; best model is re-instantiated
# with full settings before the final fit (see step 7).
CANDIDATES = {
    "Logistic Regression":    LogisticRegression(max_iter=300, solver="saga", random_state=42, n_jobs=-1),
    "Decision Tree":          DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=42),
}

# Full-quality estimators used after the best algorithm is chosen.
BEST_ESTIMATORS = {
    "Logistic Regression":    LogisticRegression(max_iter=300, solver="saga", random_state=42, n_jobs=-1),
    "Decision Tree":          DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=42),
}

print("\n[5] Comparing models (3-fold CV F1-score on training data)…\n")
results = {}
for name, clf in CANDIDATES.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
    results[name] = {"cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std(), "pipe": pipe}
    print(f"  {name:<25} CV F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 7. Select best model
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
print(f"\n[6] Best model: {best_name}")

# Rebuild pipeline with the full-quality estimator for the final fit.
best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", BEST_ESTIMATORS[best_name]),
])

# ---------------------------------------------------------------------------
# 8. Retrain best model on full training set and evaluate on test set
# ---------------------------------------------------------------------------
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print(f"\n[7] Test-set evaluation ({best_name}):")
print(f"    Accuracy : {acc:.4f}")
print(f"    F1-Score : {f1:.4f}")
print(f"    ROC-AUC  : {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

# ---------------------------------------------------------------------------
# 9. Save pipeline + metadata
# ---------------------------------------------------------------------------
artifact = {
    "pipeline": best_pipe,
    "model_name": best_name,
    "features": FEATURES,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "target": TARGET,
    "label_map": {0: "<=50K", 1: ">50K"},
    "test_accuracy": round(acc, 4),
    "test_f1": round(f1, 4),
    "test_auc": round(auc, 4),
    # Store unique values for categorical dropdowns in the Streamlit app
    "cat_unique": {col: sorted(df[col].dropna().unique().tolist()) for col in CAT_COLS},
    "num_stats": df[NUM_COLS].describe().to_dict(),
}

joblib.dump(artifact, MODEL_PATH)
print(f"\n[8] Model artifact saved -> {MODEL_PATH}")
print("=" * 60)
print("Training complete. Run:  streamlit run app.py")
print("=" * 60)
