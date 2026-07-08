import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.preprocessing import TextPreprocessor

df = pd.read_csv("data/customer_support_text_classification.csv")


X = df["customer_message"]
y = df["sentiment_label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline
pipeline = Pipeline([
    ("preprocessor", TextPreprocessor()),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500,
        min_df=2)),
    ("classifier", LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "models/sentiment_pipeline.pkl")

print("Model saved successfully!")