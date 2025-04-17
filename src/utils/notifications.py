import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import logging
import traceback

def send_alert_email(to_email, subject, message, from_email=None):
    """
    Send an alert email with robust error handling
    
    Args:
        to_email (str or list): Recipient email address(es)
        subject (str): Email subject
        message (str): Email body
        from_email (str, optional): Sender email address
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Ensure logging is configured
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure to_email is a list
        if isinstance(to_email, str):
            to_email = [to_email]
        
        # Get SMTP configuration from Streamlit secrets
        if not hasattr(st, 'secrets') or 'email' not in st.secrets:
            logger.error("No email configuration found in Streamlit secrets")
            return False
        
        smtp_config = st.secrets.email
        
        # Create message
        message_obj = MIMEMultipart('alternative')
        message_obj['Subject'] = subject
        message_obj['From'] = from_email or smtp_config.get('smtp_username', '')
        message_obj['To'] = ', '.join(to_email)
        
        # Attach message body
        html_part = MIMEText(message, 'html')
        message_obj.attach(html_part)
        
        # Connect to SMTP server
        try:
            server = smtplib.SMTP(
                smtp_config.get('smtp_host', 'smtp.gmail.com'), 
                smtp_config.get('smtp_port', 587)
            )
            
            # Enable TLS
            if smtp_config.get('use_tls', True):
                server.starttls()
            
            # Login
            server.login(
                smtp_config.get('smtp_username', ''), 
                smtp_config.get('smtp_password', '')
            )
            
            # Send email
            server.sendmail(
                message_obj['From'], 
                to_email, 
                message_obj.as_string()
            )
            
            # Close connection
            server.quit()
            
            logger.info(f"Alert email sent successfully to {', '.join(to_email)}")
            return True
        
        except smtplib.SMTPException as smtp_error:
            logger.error(f"SMTP Error: {smtp_error}")
            logger.error(traceback.format_exc())
            return False
        except Exception as e:
            logger.error(f"Email sending error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    except Exception as unexpected_error:
        logger.error(f"Unexpected error in send_alert_email: {unexpected_error}")
        logger.error(traceback.format_exc())
        return False