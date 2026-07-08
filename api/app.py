from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_sentiment

app = FastAPI(
    title="Customer Support Sentiment API",
    version="1.0.0",
    description="Predict customer support sentiment using a Scikit-learn pipeline."
)


# -----------------------------
# Request Schema
# -----------------------------

class TextRequest(BaseModel):
    text: str


# -----------------------------
# Home Route
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Support Sentiment API is running!"
    }


# -----------------------------
# Prediction Route
# -----------------------------

@app.post("/predict")
def predict(request: TextRequest):

    result = predict_sentiment(request.text)

    return result