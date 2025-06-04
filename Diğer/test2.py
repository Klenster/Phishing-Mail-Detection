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

# Initialize Typer app
app = typer.Typer()

# Download NLTK stopwords
nltk.download('stopwords')

EMAIL = "your-email@gmail.com"
PASSWORD = "your-app-password"  # ⚠️ Use an App Password instead of your real password!

# Train spam detection model
@app.command()
def train_model():
    """Train the spam detection model."""
    typer.echo("📊 Training model...")

    df = pd.read_csv('spam_ham_dataset.csv')
    df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ''))
    
    stemmer = PorterStemmer()
    corpus = []
    stopwords_set = set(stopwords.words('english'))

    for i in range(len(df)):
        text = df['text'].iloc[i].lower()
        text = text.translate(str.maketrans('', '', string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopwords_set]
        text = ' '.join(text)
        corpus.append(text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    y = df.label_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    # Save model globally
    global MODEL, VECTORIZER
    MODEL, VECTORIZER = clf, vectorizer

    typer.echo("✅ Model training completed!")

# Fetch and classify emails
@app.command()
def check_email():
    """Fetch emails and classify them as spam or not."""
    if "MODEL" not in globals() or "VECTORIZER" not in globals():
        typer.echo("⚠️ Train the model first using `python script.py train-model`")
        return

    typer.echo("📬 Checking emails...")

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, 'UNSEEN')
    email_ids = messages[0].split()

    for email_id in email_ids:
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
                    except:
                        body = ""
                    break
        else:
            try:
                body = msg.get_payload(decode=True).decode()
            except:
                body = ""

        typer.echo(f"📩 From: {msg['From']}, Subject: {subject}")

        if not body.strip():
            typer.echo("⚠️ Empty email, skipping...")
            continue

        # Preprocess email for spam detection
        stemmer = PorterStemmer()
        stopwords_set = set(stopwords.words('english'))
        email_text = body.lower().translate(str.maketrans('', '', string.punctuation)).split()
        email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
        email_text = ' '.join(email_text)

        # Transform email text and predict if it's spam
        X_email = VECTORIZER.transform([email_text]).toarray()
        prediction = MODEL.predict(X_email)

        if prediction == 1:
            typer.echo("🚨 Spam detected! Generating response...")
            generate_spam_response(msg['From'], subject)

# Generate spam response using OpenAI GPT
@app.command()
def generate_spam_response(recipient, subject):
    openai.api_key = "sk-proj-2l1auTkOSHSpYPSo1dPz5luadf1iGi45ABrOTp8Kr-kdd6v0Ka9soQCCKPbgOuXOEv_" \
    "1Vb3475T3BlbkFJVFBa_65lfckZ5JJXlUc6SXKv_hXI8Sa7GwmaKMQ4cRY_E9inWthPmTGOR6MlwKkUutssMc4doA"  
    prompt = f"Generate a spam response to the subject '{subject}'"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )

    generated_message = response.choices[0].text.strip()
    send_spam_email(recipient, generated_message)

# Send spam email response
@app.command()
def send_spam_email(recipient, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL
    msg['To'] = recipient
    msg['Subject'] = "Re: Spam"
    
    msg.attach(MIMEText(message, 'plain'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, recipient, msg.as_string())

    typer.echo(f"📩 Spam response sent to {recipient}")

# Start scheduled email checking
@app.command()
def start_scheduler():
    """Start the email checking process every 10 minutes."""
    if "MODEL" not in globals() or "VECTORIZER" not in globals():
        typer.echo("⚠️ Train the model first using `python script.py train-model`")
        return

    typer.echo("⏳ Starting scheduled email checks every 10 minutes...")

    schedule.every(10).minutes.do(check_email)

    while True:
        schedule.run_pending()
        time.sleep(1)

# Run the app
if __name__ == "__main__":
    app()
