from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="ChurnSense API", version="1.0")

model = joblib.load("model.pkl")

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: int        # 0: Month-to-month, 1: One year, 2: Two year
    InternetService: int # 0: DSL, 1: Fiber optic, 2: No
    PaymentMethod: int   # 0-3
    TechSupport: int     # 0: No, 1: Yes, 2: No internet
    OnlineSecurity: int
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    OnlineBackup: int
    DeviceProtection: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int

@app.get("/")
def root():
    return {"message": "ChurnSense API çalışıyor"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    EXPECTED_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    prob = model.predict_proba(input_df[EXPECTED_COLS])[0][1]
    risk = "Yüksek" if prob > 0.7 else "Orta" if prob > 0.4 else "Düşük"
    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": risk,
        "recommendation": "Müşteriyle acil iletişime geç" if risk == "Yüksek" else "Takipte kal" if risk == "Orta" else "Müşteri güvende"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
    