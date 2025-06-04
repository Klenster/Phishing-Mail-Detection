import string
import numpy as np
import pandas as pd
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time

# Email credentials and IMAP setup
EMAIL = "arapnecmi2@gmail.com"
PASSWORD = "stxesiscqfnuzgqk"

# Spam detection setup
def spam_detection(): 
    df = pd.read_csv('spam_ham_dataset.csv')  # Load dataset
    df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ''))  # Clean up newline characters
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    
    return clf, vectorizer

# Fetch emails and classify them as spam or not
def fetch_and_classify_emails(clf, vectorizer):
    # Connect to Gmail IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")
    
    # Search for unread emails
    status, messages = mail.search(None, 'UNSEEN')
    email_ids = messages[0].split()
    
    for email_id in email_ids:
        _, msg_data = mail.fetch(email_id, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        
        # Extract email content
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()

        print(f"From: {msg['From']}, Subject: {subject}")
        stemmer = PorterStemmer()
        corpus = []
        stopwords_set = set(stopwords.words('english'))

        # Preprocess email for spam detection
        email_text = body.lower().translate(str.maketrans('', '', string.punctuation)).split()
        email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
        email_text = ' '.join(email_text)

        # Transform email text and predict if it's spam
        email_corpus = [email_text]
        X_email = vectorizer.transform(email_corpus).toarray()
        prediction = clf.predict(X_email)

        if prediction == 1:  # Spam detected
            print("Spam detected, generating response...")
            generate_spam_response(msg['From'], subject)

# Use OpenAI GPT to generate spam response
def generate_spam_response(recipient, subject):
    openai.api_key = "your-openai-api-key"
    
    prompt = f"Generate a spam response to the subject '{subject}'"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )

    generated_message = response.choices[0].text.strip()

    # Send the generated spam response to the sender
    send_spam_email(recipient, generated_message)

# Send the spam email response
def send_spam_email(recipient, message):
    msg = MIMEMultipart()
    msg['From'] = EMAIL
    msg['To'] = recipient
    msg['Subject'] = "Re: Spam"
    
    msg.attach(MIMEText(message, 'plain'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL, PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL, recipient, text)
        print(f"Spam response sent to {recipient}")

# Schedule the email checking process
def schedule_email_checking(clf, vectorizer):
    schedule.every(10).minutes.do(fetch_and_classify_emails, clf=clf, vectorizer=vectorizer)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Train spam detection model
    clf, vectorizer = spam_detection()
    
    # Start scheduling email checks
    schedule_email_checking(clf, vectorizer)
