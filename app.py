# delinquency_risk_dashboard.py
"""
Delinquency-Risk Scoring Dashboard  📊📉
────────────────────────────────────────────────────────────────────────────
**POC workflow**

1. Upload a CSV of *current delinquent* loans.  
   – Include a historical “resolved” (1/0) column so we can train a quick model.  
2. Pick the target & basic cost/benefit assumptions.  
3. App trains a Gradient-Boosting classifier ➜ scores every loan.  
4. Loans are bucketed into **deciles** (10 = most likely to resolve).  
5. Dashboard shows cohort KPIs and a slider to simulate which top-N deciles
   collection agents should work to maximize net benefit.

> Demo-level only — no production data, auth, or model governance.
> For enterprise implementations, reach out: https://drtomharty.com/bio
"""
# ────────────────────────────────────────────────── imports ───────────────
import os, pickle
import pandas as pd, numpy as np, streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ─────────────────────────── preprocessor / helper ────────────────────────
def build_pipeline(df, target):
    num_cols = df.drop(columns=[target]).select_dtypes(include="number").columns
    cat_cols = df.drop(columns=[target]).select_dtypes(exclude="number").columns

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc" , StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols),
                             ("cat", cat_pipe, cat_cols)])
    clf = GradientBoostingClassifier(random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])

def make_deciles(prob, n=10):
    # higher prob => higher decile number
    return pd.qcut(prob.rank(method="first"), q=n, labels=False) + 1

# ─────────────────────────────–– UI  ──────────────────────────────────────
st.set_page_config(page_title="Delinquency-Risk Scoring", layout="wide")
st.title("💸 Delinquency-Risk Scoring Dashboard")

st.sidebar.info(
    "💡 **Demo Notice**\n"
    "This proof-of-concept trains a small model live. "
    "For regulated, production-grade workflows (MLOps, bias testing, audit "
    "trails), [contact me](https://drtomharty.com/bio)."
)

data_file = st.sidebar.file_uploader("📂 Upload delinquent-loan CSV", type="csv")
if not data_file:
    st.warning("Upload a CSV to begin.", icon="📄")
    st.stop()

df = pd.read_csv(data_file)
st.subheader("Data preview")
st.dataframe(df.head())

target = st.selectbox("🎯 Historical resolution column (1 = resolved)", df.columns)
cost_per_account = st.number_input("💰 Cost to work one account ($)", 1.0, 100.0, 5.0)
recovery_per_account = st.number_input("💵 Benefit if account resolves ($)", 10.0, 500.0, 50.0)
test_size = st.slider("Validation split %", 0.1, 0.4, 0.2, 0.05)

if st.button("🚀 Train & Score"):
    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=[target]), df[target],
        test_size=test_size, stratify=df[target], random_state=42
    )
    pipe = build_pipeline(df, target)
    with st.spinner("Training model…"):
        pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(df.drop(columns=[target]))[:, 1]
    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])

    scored = df.copy()
    scored["resolve_prob"] = proba
    scored["decile"] = make_deciles(proba)

    # cohort summary
    summary = (scored.groupby("decile")
               .agg(accounts=("decile", "count"),
                    avg_prob=("resolve_prob", "mean"))
               .sort_index(ascending=False)
               .reset_index())

    st.success(f"Model trained! Validation ROC-AUC: **{auc:.3f}**")
    st.subheader("Decile Summary (10 = best)")
    st.dataframe(summary.style.format({"avg_prob":"{:.2%}"}))

    # bar plot
    fig = px.bar(summary, x="decile", y="avg_prob",
                 labels={"avg_prob":"Avg. resolve probability"},
                 title="Average Resolve Probability by Decile")
    st.plotly_chart(fig, use_container_width=True)

    # benefit simulator
    st.subheader("📈 Workforce Simulation")
    top_n = st.slider("Work top … deciles (10 = only best decile, 1 = all)", 1, 10, 3)
    work_mask = scored["decile"] >= (11 - top_n)
    worked = scored[work_mask]

    expected_resolves = worked["resolve_prob"].sum()
    total_cost  = worked.shape[0] * cost_per_account
    total_gain  = expected_resolves * recovery_per_account
    net_benefit = total_gain - total_cost

    st.metric("Accounts worked", worked.shape[0])
    st.metric("Expected resolves", f"{expected_resolves:.0f}")
    st.metric("Net benefit ($)", f"{net_benefit:,.0f}")

    # download buttons
    st.download_button("⬇️ Scored CSV",
                       data=scored.to_csv(index=False).encode(),
                       file_name="scored_loans.csv",
                       mime="text/csv")
    st.download_button("💾 Model (.pkl)",
                       data=pickle.dumps(pipe),
                       file_name="risk_model.pkl",
                       mime="application/octet-stream")
