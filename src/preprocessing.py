import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Required for newer NLTK versions
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):

        stop_words = set(stopwords.words("english"))

        negation_words = {
            "not", "no", "nor", "never",
            "don't", "didn't", "won't",
            "can't", "isn't", "wasn't",
            "aren't", "couldn't", "shouldn't"
        }

        self.stop_words = stop_words - negation_words

    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_text= []

        for text in X:

            text = text.lower()

            text = re.sub(r'[^a-zA-Z\s]', '', text)

            tokens =word_tokenize(text)
            tokens =[word for word in tokens if word not in self.stop_words]

            cleaned_text.append(" ".join(tokens))

        return cleaned_text