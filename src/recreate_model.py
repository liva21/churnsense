import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
import os

print("Veri yükleniyor...")
df = pd.read_csv('../data/processed/telco_cleaned.csv')
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

print("Kategorik değişkenler encode ediliyor...")
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

print("Model eğitiliyor...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    random_state=42,
    eval_metric='auc'
)
model.fit(X, y)

print("Model kaydediliyor...")
os.makedirs('api', exist_ok=True)
os.makedirs('dashboard', exist_ok=True)

joblib.dump(model, 'api/model.pkl')
joblib.dump(model, 'dashboard/model.pkl')
print("Başarılı! model.pkl hem api hem de dashboard klasörüne eklendi.")
