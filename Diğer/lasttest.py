import os
import typer
import string
import numpy as np
import pandas as pd
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time
import joblib
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Typer app
app = typer.Typer()

# Download NLTK stopwords (only once)
nltk.download('stopwords', quiet=True)

# Load credentials from environment variables
EMAIL = "arapnecmi2@gmail.com"
PASSWORD =  "stxe sisc qfnu zgqk" 
OPENAI_API_KEY = "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"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and vectorizer
MODEL = None
VECTORIZER = None

# Preprocess text for spam detection
def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, stopwords, and stemming."""
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

# Train spam detection model
@app.command()
def train_model():
    """Train the spam detection model and save it to disk."""
    logger.info("📊 Training model...")

    try:
        df = pd.read_csv('spam_ham_dataset.csv')
        df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ''))

        corpus = df['text'].apply(preprocess_text)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus).toarray()
        y = df.label_num

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(X_train, y_train)

        # Save model and vectorizer to disk
        joblib.dump(clf, 'spam_model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        logger.info("✅ Model training completed and saved to disk!")
    except Exception as e:
        logger.error(f"⚠️ Failed to train model: {e}")

# Load model and vectorizer from disk
@app.command()
def load_model():
    """Load the trained model and vectorizer from disk."""
    global MODEL, VECTORIZER
    try:
        MODEL = joblib.load('spam_model.pkl')
        VECTORIZER = joblib.load('vectorizer.pkl')
        logger.info("✅ Model and vectorizer loaded successfully!")
    except Exception as e:
        logger.error(f"⚠️ Failed to load model: {e}")

# Fetch and classify emails
@app.command()
def check_email():
    """Fetch emails and classify them as spam or not."""
    if MODEL is None or VECTORIZER is None:
        logger.warning("⚠️ Train the model first using `python script.py train-model`")
        return

    logger.info("📬 Checking emails...")

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")

        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()

        for email_id in email_ids:
            try:
                _, msg_data = mail.fetch(email_id, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8")

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body = part.get_payload(decode=True).decode()
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to decode multipart email: {e}")
                            break
                else:
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to decode email: {e}")

                logger.info(f"📩 From: {msg['From']}, Subject: {subject}")

                if not body.strip():
                    logger.warning("⚠️ Empty email, skipping...")
                    continue

                # Preprocess email for spam detection
                email_text = preprocess_text(body)

                # Transform email text and predict if it's spam
                X_email = VECTORIZER.transform([email_text]).toarray()
                prediction = MODEL.predict(X_email)

                if prediction == 1:
                    logger.info("🚨 Spam detected! Generating response...")
                    generate_spam_response(msg['From'], subject)
            except Exception as e:
                logger.error(f"⚠️ Failed to process email: {e}")
    except Exception as e:
        logger.error(f"⚠️ Failed to fetch emails: {e}")
    finally:
        mail.logout()

# Generate spam response using OpenAI GPT
@app.command()
def generate_spam_response(recipient, subject):
    """Generate a response to a spam email using OpenAI GPT."""
    try:
        openai.api_key = OPENAI_API_KEY

        prompt = f"Generate a polite but firm response to the email with subject '{subject}' indicating that it has been flagged as spam."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100
        )

        generated_message = response.choices[0].text.strip()
        send_spam_email(recipient, generated_message,subject)
    except Exception as e:
        logger.error(f"⚠️ Failed to generate response: {e}")

# Send spam email response
@app.command()
def send_spam_email(recipient, message,subject):
    """Send a response email to the spam sender."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = recipient
        msg['Subject'] = f"Re: {subject}"

        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL, PASSWORD)
            server.sendmail(EMAIL, recipient, msg.as_string())

        logger.info(f"📩 Spam response sent to {recipient}")
    except Exception as e:
        logger.error(f"⚠️ Failed to send email: {e}")

# Start scheduled email checking
@app.command()
def start_scheduler():
    """Start the email checking process every 10 minutes."""
    if MODEL is None or VECTORIZER is None:
        logger.warning("⚠️ Train the model first using `python script.py train-model`")
        return

    logger.info("⏳ Starting scheduled email checks every 10 minutes...")

    schedule.every(10).minutes.do(check_email)

    while True:
        schedule.run_pending()
        time.sleep(1)

# Run the app
if __name__ == "__main__":
    load_model()  # Load model and vectorizer at startup
    app()