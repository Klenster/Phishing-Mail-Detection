import os
from dotenv import load_dotenv
load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SPAM_MODEL_PATH      = os.getenv("SPAM_MODEL_PATH", "spam_model.pkl")
VECTORIZER_MODEL_PATH= os.getenv("VECTORIZER_MODEL_PATH", "vectorizer.pkl")