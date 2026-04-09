import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ChurnSense", page_icon="🔮", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("../../data/processed/telco_cleaned.csv")

model = load_model()
df = load_data()

st.title("🔮 ChurnSense — Müşteri Kayıp Tahmin Platformu")
st.markdown("XGBoost + SHAP ile açıklanabilir müşteri churn analizi")

tab1, tab2, tab3 = st.tabs(["📊 Genel Bakış", "🔍 Tahmin", "🧠 SHAP Analizi"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Müşteri", len(df))
    col2.metric("Churn Oranı", f"{df['Churn'].mean()*100:.1f}%")
    col3.metric("Ort. Tenure", f"{df['tenure'].mean():.0f} ay")

    st.subheader("Churn Dağılımı")
    st.image("../../data/processed/churn_distribution.png")

    col4, col5 = st.columns(2)
    with col4:
        st.image("../../data/processed/tenure_churn.png")
    with col5:
        st.image("../../data/processed/monthly_charges_churn.png")

with tab2:
    st.subheader("Müşteri Churn Tahmini")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (ay)", 0, 72, 12)
        monthly = st.slider("Aylık Ücret ($)", 18, 120, 65)
        total = st.number_input("Toplam Ücret ($)", value=float(monthly * tenure))
    with col2:
        contract = st.selectbox("Sözleşme", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("İnternet Servisi", ["DSL", "Fiber optic", "No"])
        tech = st.selectbox("Teknik Destek", ["No", "Yes", "No internet service"])
    with col3:
        security = st.selectbox("Online Güvenlik", ["No", "Yes", "No internet service"])
        payment = st.selectbox("Ödeme Yöntemi", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        paperless = st.selectbox("Kağıtsız Fatura", ["Yes", "No"])

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    tech_map = {"No": 0, "Yes": 1, "No internet service": 2}
    payment_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}

    input_data = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract_map[contract], "InternetService": internet_map[internet],
        "TechSupport": tech_map[tech], "OnlineSecurity": tech_map[security],
        "PaymentMethod": payment_map[payment], "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
        "PhoneService": 1, "MultipleLines": 0, "OnlineBackup": 0,
        "DeviceProtection": 0, "StreamingTV": 0, "StreamingMovies": 0
    }])

    EXPECTED_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    if st.button("🔮 Tahmin Et", type="primary"):
        prob = model.predict_proba(input_data[EXPECTED_COLS])[0][1]
        risk = "🔴 Yüksek Risk" if prob > 0.7 else "🟡 Orta Risk" if prob > 0.4 else "🟢 Düşük Risk"
        st.metric("Churn Olasılığı", f"{prob*100:.1f}%")
        
        if prob > 0.7:
            st.error(risk)
            st.info("💡 Öneri: Müşteriyle acil iletişime geç, özel indirim teklif et.")
        elif prob > 0.4:
            st.warning(risk)
            st.info("💡 Öneri: Müşteriyi takibe al, memnuniyet anketi gönder.")
        else:
            st.success(risk)
            st.info("💡 Öneri: Müşteri güvende, upsell fırsatlarını değerlendir.")

with tab3:
    st.subheader("SHAP Feature Importance")
    st.image("../../data/processed/shap_importance.png")
    st.subheader("SHAP Summary Plot")
    st.image("../../data/processed/shap_summary.png")
    
