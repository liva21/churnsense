# 🔮 ChurnSense — Customer Churn Prediction Platform

Real-time customer churn prediction for SaaS companies using XGBoost and Explainable AI (SHAP).

## What it does
- Predicts churn probability for each customer (0-100%)
- Explains *why* a customer is at risk using SHAP values
- Provides actionable recommendations for sales & CS teams
- REST API endpoint for integration with CRM systems

## Tech Stack
- **ML:** XGBoost, LightGBM, Scikit-learn
- **Explainability:** SHAP
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Data:** IBM Telco Customer Churn Dataset (7,043 customers)

## Results
- ROC-AUC: ~0.85
- Identifies 80%+ of churners before they leave

## Quick Start
```bash
git clone https://github.com/liva21/churnsense.git
cd churnsense
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard.py
```

## Project Structure
```
churnsense/
├── data/
│   ├── raw/          # Original dataset
│   └── processed/    # Cleaned data & visualizations
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model.ipynb
└── src/
    ├── api.py        # FastAPI endpoint
    └── dashboard.py  # Streamlit UI
```
