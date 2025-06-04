import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .config import IMAP_SERVER, EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT
import logging

logger = logging.getLogger(__name__)

class EmailClient:
    def __init__(self):
        # IMAP bağlantısı
        self.imap = imaplib.IMAP4_SSL(IMAP_SERVER)
        self.imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

    def fetch_unseen(self, mailbox="INBOX"):
        """Görüntülenmemiş (UNSEEN) e-postaları getirir."""
        self.imap.select(mailbox)
        _, data = self.imap.search(None, 'UNSEEN')
        email_ids = data[0].split()
        messages = []
        for eid in email_ids:
            _, msg_data = self.imap.fetch(eid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            messages.append((eid, msg))
        return messages

    def mark_seen(self, eid):
        """Bir e-postayı SEEN olarak işaretler."""
        self.imap.store(eid, '+FLAGS', '\\Seen')

    def send_email(self, to_address: str, subject: str, body: str):
        """SMTP ile e-posta gönderir."""
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To']   = to_address
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            logger.info(f"Sent email to {to_address}")
