"""
app.py
------
Streamlit web application for Adult Income (Salary) Prediction.
Loads the trained sklearn Pipeline from models/model.pkl and serves
real-time predictions from user-provided inputs.

Run:
    streamlit run app.py
"""

import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model artifact (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML model…")
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #57606a;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #57606a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f2328;
    }
    .predict-btn > button {
        background-color: #3b82d4 !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        width: 100% !important;
        padding: 0.6rem !important;
    }
    .result-box-high {
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-box-low {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-label {
        font-size: 1.1rem;
        color: #57606a;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }
    .confidence-bar-wrap {
        background: #e5e7eb;
        border-radius: 999px;
        height: 12px;
        margin-top: 0.8rem;
    }
    .divider { border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">💼 Adult Income Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict whether annual income is <strong>&gt;50K</strong> or <strong>≤50K</strong> '
    'using a trained Machine Learning model on the UCI Adult Census dataset.</div>',
    unsafe_allow_html=True,
)

if artifact is None:
    st.error(
        "**Model not found.**  "
        "Please run `python train_model.py` first to train and save the model."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"**Algorithm:** {artifact['model_name']}")
    st.markdown(f"**Target:** `{artifact['target']}`")
    st.markdown("---")
    st.markdown("**Test-set Metrics**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", f"{artifact['test_accuracy']:.2%}")
        st.metric("F1-Score", f"{artifact['test_f1']:.4f}")
    with col_b:
        st.metric("ROC-AUC", f"{artifact['test_auc']:.4f}")

    st.markdown("---")
    st.markdown("**Features used**")
    for f in artifact["features"]:
        st.markdown(f"• `{f}`")
    st.markdown("---")
    st.caption("Model: `models/model.pkl`")
    st.caption("Data: `data/adult.csv`")

# ---------------------------------------------------------------------------
# Main — input form
# ---------------------------------------------------------------------------
st.subheader("Enter Individual Details")
st.markdown("Fill in the fields below and click **Predict Income** to get a result.")

cat_unique: dict = artifact["cat_unique"]
num_stats: dict = artifact["num_stats"]
num_cols: list = artifact["num_cols"]
cat_cols: list = artifact["cat_cols"]


def _stat(col, stat, fallback):
    try:
        return float(num_stats[col][stat])
    except Exception:
        return fallback


with st.form("prediction_form"):
    # ---- Row 1: Demographics ------------------------------------------------
    st.markdown("##### 👤 Demographics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input(
            "Age",
            min_value=17, max_value=90,
            value=int(_stat("age", "50%", 38)),
            step=1,
        )
    with c2:
        gender = st.selectbox("Gender", options=cat_unique.get("gender", ["Male", "Female"]))
    with c3:
        race = st.selectbox("Race", options=cat_unique.get("race", ["White"]))
    with c4:
        native_country = st.selectbox(
            "Native Country", options=cat_unique.get("native-country", ["United-States"])
        )

    # ---- Row 2: Education ---------------------------------------------------
    st.markdown("##### 🎓 Education")
    c1, c2 = st.columns(2)
    with c1:
        education = st.selectbox(
            "Education Level", options=cat_unique.get("education", ["HS-grad"])
        )
    with c2:
        educational_num = st.slider(
            "Education (numeric)",
            min_value=1, max_value=16,
            value=int(_stat("educational-num", "50%", 9)),
            help="1 = Preschool → 16 = Doctorate",
        )

    # ---- Row 3: Work --------------------------------------------------------
    st.markdown("##### 💼 Employment")
    c1, c2, c3 = st.columns(3)
    with c1:
        workclass = st.selectbox("Work Class", options=cat_unique.get("workclass", ["Private"]))
    with c2:
        occupation = st.selectbox("Occupation", options=cat_unique.get("occupation", ["Prof-specialty"]))
    with c3:
        hours_per_week = st.number_input(
            "Hours per Week",
            min_value=1, max_value=99,
            value=int(_stat("hours-per-week", "50%", 40)),
            step=1,
        )

    # ---- Row 4: Family ------------------------------------------------------
    st.markdown("##### 👨‍👩‍👧 Family & Relationship")
    c1, c2 = st.columns(2)
    with c1:
        marital_status = st.selectbox(
            "Marital Status", options=cat_unique.get("marital-status", ["Married-civ-spouse"])
        )
    with c2:
        relationship = st.selectbox(
            "Relationship", options=cat_unique.get("relationship", ["Husband"])
        )

    # ---- Row 5: Financial ---------------------------------------------------
    st.markdown("##### 💰 Financial")
    c1, c2 = st.columns(2)
    with c1:
        capital_gain = st.number_input(
            "Capital Gain ($)",
            min_value=0, max_value=99999,
            value=0, step=100,
        )
    with c2:
        capital_loss = st.number_input(
            "Capital Loss ($)",
            min_value=0, max_value=4356,
            value=0, step=100,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔮  Predict Income", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "education": education,
        "educational-num": educational_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }])

    # Ensure column order matches training
    input_data = input_data[artifact["features"]]

    pipeline = artifact["pipeline"]
    pred_class = int(pipeline.predict(input_data)[0])
    pred_proba = pipeline.predict_proba(input_data)[0]
    confidence = float(pred_proba[pred_class]) * 100
    prob_high = float(pred_proba[1]) * 100
    prob_low  = float(pred_proba[0]) * 100

    label_map = artifact["label_map"]
    predicted_label = label_map[pred_class]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    res_col, prob_col = st.columns([1, 1])

    with res_col:
        if pred_class == 1:
            box_class = "result-box-high"
            icon = "✅"
        else:
            box_class = "result-box-low"
            icon = "⚠️"

        st.markdown(
            f'<div class="{box_class}">'
            f'<div class="result-label">Predicted Income</div>'
            f'<div class="result-value">{icon} {predicted_label}</div>'
            f'<div style="color:#57606a;margin-top:0.6rem;">Confidence: <strong>{confidence:.1f}%</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with prob_col:
        st.markdown("**Class Probabilities**")
        st.markdown(f"🟢 &nbsp; **>50K** &nbsp; — &nbsp; `{prob_high:.1f}%`")
        st.progress(int(prob_high))
        st.markdown(f"🟡 &nbsp; **≤50K** &nbsp; — &nbsp; `{prob_low:.1f}%`")
        st.progress(int(prob_low))

    with st.expander("📋 Input summary (what was sent to the model)"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with **Streamlit** · **scikit-learn** · **pandas**  |  "
    f"Model: **{artifact['model_name']}**  |  "
    "Dataset: UCI Adult Census Income"
)
