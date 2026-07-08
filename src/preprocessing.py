import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
negation_words = {
               "not", "no", "nor", "never",
                "don't", "didn't", "won't",
                 "can't", "isn't", "wasn't",
               "aren't", "couldn't", "shouldn't"}
stop_words = stop_words - negation_words

class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_text= []

        for text in X:

            text = text.lower()

            text = re.sub(r'[^a-zA-Z\s]', '', text)

            tokens =word_tokenize(text)
            tokens =[word for word in tokens if word not in stop_words]

            cleaned_text.append(" ".join(tokens))

        return cleaned_text