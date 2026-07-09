# Business Problem

Customer support teams receive thousands of customer messages every day.

Manually categorizing customer feedback is time-consuming and difficult to scale.

This project automates sentiment classification so organizations can:

Prioritize negative customer complaints
Monitor customer satisfaction
Analyze customer feedback trends
Improve response time
Support customer experience analytics

# Dataset

Dataset contains customer support messages labeled into three sentiment classes.

Classes:

Positive
Neutral
Negative

# Features
Text preprocessing
Custom Scikit-Learn Transformer
TF-IDF Vectorization
Logistic Regression classifier
Scikit-Learn Pipeline
Probability prediction
Confidence score
REST API using FastAPI
Interactive Streamlit Web App
Cloud Deployment
Model Serialization using Joblib

# Machine Learning Pipeline

The prediction pipeline consists of:

Lowercasing text
Removing special characters
Stopword removal
Preserving negation words
TF-IDF Vectorization
Logistic Regression Classification

Entire preprocessing and prediction workflow is encapsulated using a Scikit-Learn Pipeline.

# Tech Stack
Programming
Python
Machine Learning
Scikit-Learn
TF-IDF
Logistic Regression
NLP
NLTK
Regular Expressions
Backend
FastAPI
Uvicorn
Frontend
Streamlit
Deployment
Render
Streamlit Cloud
Version Control
Git
GitHub

# Deployment
Backend

Render

Live API
https://customer-support-api-freb.onrender.com

Streamlit Cloud
https://customer-support-sentimentgit-pm9cnkpz89yoehbu7ukrxn.streamlit.app/

# Note:
 This project is deployed using the free tiers of Render and Streamlit Community Cloud. The first request may take 30–60 seconds if the backend is waking up after inactivity.