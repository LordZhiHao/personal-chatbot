import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

# Create message
msg = MIMEMultipart()
msg['From'] = EMAIL_USER
msg['To'] = EMAIL_RECIPIENT
msg['Subject'] = "Test Email from Chatbot Application"

# Email body
body = """
<h2>Test Email</h2>
<p>This is a test email from your chatbot application.</p>
"""

msg.attach(MIMEText(body, 'html'))

# Connect to server and send email
try:
    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
    print("Email sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")