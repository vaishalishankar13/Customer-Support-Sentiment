import joblib

from src.preprocessing import TextPreprocessor



from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.pkl"




# Loading model only once
pipeline = joblib.load(MODEL_PATH)


def predict_sentiment(text):
    """
    Predict sentiment for a single sentence.
    """

    prediction = pipeline.predict([text])[0]

    probabilities = pipeline.predict_proba([text])[0]

    confidence = max(probabilities)

    class_probabilities = {
        label: float(prob)
        for label, prob in zip(pipeline.classes_, probabilities)
    }

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probabilities": class_probabilities
    }


