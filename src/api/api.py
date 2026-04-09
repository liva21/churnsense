from fastapi import FastAPI
import uvicorn

app = FastAPI(title="ChurnSense API")

@app.get("/")
def read_root():
    return {"message": "Welcome to ChurnSense API"}

# @app.post("/predict")
# def predict(data: dict):
#     # Model intput parsing and prediction logic
#     pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
