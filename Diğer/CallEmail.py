import imaplib
import email
from email.header import decode_header

EMAIL = "arapnecmi2@gmail.com"
PASSWORD = "stxe sisc qfnu zgqk"

# Connect to Gmail IMAP server
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(EMAIL, PASSWORD)
mail.select("[Gmail]/Spam")

# Search for unread emails
status, messages = mail.search(None, 'UNSEEN')
print(messages)

email_ids = messages[0].split()

# Fetch each email
for email_id in email_ids:
    _, msg_data = mail.fetch(email_id, "(RFC822)")
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)
     

    # Extract metadata
    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")

    print(f"From: {msg['From']}, Subject: {subject}")