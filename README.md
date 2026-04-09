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

<img width="2720" height="1592" alt="image" src="https://github.com/user-attachments/assets/14cc8167-50da-4d72-bac1-e0e5883eadc4" />
<img width="2970" height="1084" alt="image" src="https://github.com/user-attachments/assets/ee4afc23-579b-46b3-ac45-757ee2145a65" />
<img width="1518" height="1564" alt="image" src="https://github.com/user-attachments/assets/6e4cc61a-198c-45a2-b1a8-29822af4d237" />
<img width="1340" height="1330" alt="image" src="https://github.com/user-attachments/assets/9fdb2717-b055-47cd-be18-2e97209c7af8" />



