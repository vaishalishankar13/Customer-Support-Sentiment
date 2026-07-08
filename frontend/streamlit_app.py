import streamlit as st
import requests

import os

API_URL=os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)


st.set_page_config(
    page_title="Customer Support Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Customer Support Sentiment Analyzer")

st.write(
    "Enter a customer support message below and click **Predict Sentiment**."
)

text = st.text_area(
    "Customer Message",
    height=150,
    placeholder="Type your message here..."
)

if st.button("Predict Sentiment"):

    if text.strip() == "":
        st.warning("Please enter a message.")

    else:

        response = requests.post(
            API_URL,
            json={"text": text}
        )

        if response.status_code == 200:

            result = response.json()

            sentiment = result["prediction"]
            confidence = result["confidence"]

            if sentiment == "positive":
                st.success(f"😊 Sentiment: {sentiment.title()}")

            elif sentiment == "negative":
                st.error(f"😠 Sentiment: {sentiment.title()}")

            else:
                st.info(f"😐 Sentiment: {sentiment.title()}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.subheader("Prediction Probabilities")

            probabilities = result["probabilities"]

            st.progress(probabilities["positive"])
            st.write(f"Positive : {probabilities['positive']:.2%}")

            st.progress(probabilities["neutral"])
            st.write(f"Neutral : {probabilities['neutral']:.2%}")

            st.progress(probabilities["negative"])
            st.write(f"Negative : {probabilities['negative']:.2%}")

        else:
            st.error("API is not running.")