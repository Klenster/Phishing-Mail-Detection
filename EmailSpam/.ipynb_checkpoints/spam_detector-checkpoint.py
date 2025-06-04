import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from EmailSpam.config import SPAM_MODEL_PATH, VECTORIZER_MODEL_PATH

class SpamDetector:
    def __init__(self):

        # Model ve vectorizer'ı yükle
        self.model = joblib.load(SPAM_MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_MODEL_PATH)

    def preprocess_text(self, text: str) -> str:
        """Lowercase, strip punctuation, remove stopwords, stem."""
        import string, nltk
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer

        nltk.download('stopwords', quiet=True)
        stemmer = PorterStemmer()
        stops = set(stopwords.words('english'))

        tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        stems = [stemmer.stem(w) for w in tokens if w not in stops]
        return ' '.join(stems)

    def predict(self, text: str) -> dict:
        """Metni sınıflandır ve olasılık döndür."""
        cleaned = self.preprocess_text(text)
        X = self.vectorizer.transform([cleaned])
        label = self.model.predict(X)[0]
        proba = float(self.model.predict_proba(X)[0][label])
        return {"label": int(label), "confidence": proba}
