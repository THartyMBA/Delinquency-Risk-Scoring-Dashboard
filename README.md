# Delinquency-Risk-Scoring-Dashboard

💸 Delinquency-Risk Scoring Dashboard
Upload a CSV of delinquent loans ➜ train a Gradient-Boosting model in seconds ➜ see 10-decile risk buckets and simulate the $$$ impact of assigning collectors to the most promising accounts.

🔍 What the app does

Step	Action
1	Upload a CSV that contains:
• borrower-level features
• a binary column (1 = resolved / cured, 0 = still delinquent) for historical training
2	Auto-detects numeric vs. categorical columns and builds a pipeline (impute ▸ scale ▸ one-hot ▸ GradientBoosting).
3	Splits data into train / validation (stratified) and reports ROC-AUC.
4	Scores the entire file, bins probabilities into 10 deciles (10 = most likely to resolve).
5	Interactive bar chart of average resolve-probability per decile.
6	Work-force simulator – a slider lets you choose how many top-N deciles agents will work; shows expected resolves, total cost, and net benefit.
7	One-click download of scored_loans.csv and the trained risk_model.pkl.
Proof-of-concept only – no hyper-parameter tuning, fairness diagnostics, or MLOps.
Need an enterprise scorecard pipeline? → drtomharty.com/bio



🛠️ Requirements
nginx
Copy
Edit
streamlit
pandas
numpy
plotly
scikit-learn
(All CPU wheels – runs on Streamlit Cloud’s free tier.)

🚀 Quick start (local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/delinquency-risk-dashboard.git
cd delinquency-risk-dashboard
python -m venv venv && source venv/bin/activate     # Win: venv\Scripts\activate
pip install -r requirements.txt
streamlit run delinquency_risk_dashboard.py
Open http://localhost:8501, upload your CSV, and explore.

☁️ Deploy free on Streamlit Cloud
Push the repo (public or private) to GitHub.

Go to streamlit.io/cloud ➜ New app, select repo/branch, click Deploy.

Share the public URL with stakeholders.

No secrets or API keys required.

🗂️ Repo layout
kotlin
Copy
Edit
delinquency_risk_dashboard.py   ← single-file app
requirements.txt
README.md                        ← this file
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – fast, interactive UIs

scikit-learn – GradientBoosting & preprocessing

Plotly – beautiful interactive charts

Forecast cost savings, optimize collections, iterate – enjoy! 🎉
