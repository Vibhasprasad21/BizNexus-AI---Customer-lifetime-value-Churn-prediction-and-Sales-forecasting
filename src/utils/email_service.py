import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from firebase_admin import firestore
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_smtp_credentials(user_id=None):
    """
    Get SMTP credentials with priority:
    1. Streamlit secrets (system-wide fallback)
    2. User-specific configuration
    3. Company configuration
    """
    try:
        # FIRST, check Streamlit secrets - always use this if available
        if hasattr(st, 'secrets') and 'email' in st.secrets:
            logger.info("Using system-wide SMTP configuration from secrets.toml")
            return {
                'smtp_host': st.secrets.email.get('smtp_host', 'smtp.gmail.com'),
                'smtp_port': st.secrets.email.get('smtp_port', 587),
                'smtp_username': st.secrets.email.get('smtp_username', ''),
                'smtp_password': st.secrets.email.get('smtp_password', ''),
                'use_tls': st.secrets.email.get('use_tls', True),
                'configured': True
            }
        
        # If no user_id provided, return None
        if not user_id:
            logger.warning("No user_id provided and no system-wide SMTP configuration")
            return None
        
        # Initialize Firestore
        db = firestore.client()
        user_doc = db.collection('users').document(user_id).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            email_config = user_data.get('email_config', {})
            
            # Check user-specific email configuration
            if email_config.get('configured', False):
                logger.info("Using user-specific SMTP configuration")
                return {
                    'smtp_host': email_config.get('smtp_host', 'smtp.gmail.com'),
                    'smtp_port': email_config.get('smtp_port', 587),
                    'smtp_username': email_config.get('smtp_username', ''),
                    'smtp_password': email_config.get('smtp_password', ''),
                    'use_tls': email_config.get('use_tls', True)
                }
            
            # Try company SMTP configuration
            company_id = user_data.get('company_id')
            if company_id:
                company_doc = db.collection('companies').document(company_id).get()
                if company_doc.exists:
                    company_data = company_doc.to_dict()
                    company_email_config = company_data.get('email_config', {})
                    
                    if company_email_config.get('configured', False):
                        logger.info("Using company-specific SMTP configuration")
                        return {
                            'smtp_host': company_email_config.get('smtp_host', 'smtp.gmail.com'),
                            'smtp_port': company_email_config.get('smtp_port', 587),
                            'smtp_username': company_email_config.get('smtp_username', ''),
                            'smtp_password': company_email_config.get('smtp_password', ''),
                            'use_tls': company_email_config.get('use_tls', True)
                        }
        
        logger.error(f"No SMTP configuration found for user {user_id}")
        return None
    
    except Exception as e:
        logger.error(f"Error retrieving SMTP credentials: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def send_email(recipients, subject, html_content, user_id=None):
    """
    Send an email with robust error handling
    """
    try:
        # Get SMTP configuration
        smtp_config = get_smtp_credentials(user_id)
        
        if not smtp_config:
            logger.error("No SMTP configuration available")
            return False
        
        # Validate SMTP configuration
        required_fields = ['smtp_host', 'smtp_port', 'smtp_username', 'smtp_password']
        if not all(smtp_config.get(field) for field in required_fields):
            logger.error("Incomplete SMTP configuration")
            return False
        
        logger.info(f"Attempting to send email via {smtp_config['smtp_host']}:{smtp_config['smtp_port']}")
        logger.info(f"From: {smtp_config['smtp_username']}")
        logger.info(f"To: {', '.join(recipients)}")
        
        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = smtp_config['smtp_username']
        message['To'] = ', '.join(recipients)
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        message.attach(html_part)
        
        # Connect to SMTP server
        try:
            server = smtplib.SMTP(smtp_config['smtp_host'], smtp_config['smtp_port'])
            
            # Enable logging for SMTP
            server.set_debuglevel(1)
            
            if smtp_config.get('use_tls', True):
                server.starttls()
            
            # Login
            try:
                server.login(smtp_config['smtp_username'], smtp_config['smtp_password'])
            except smtplib.SMTPAuthenticationError as auth_error:
                logger.error(f"SMTP Authentication Error: {auth_error}")
                return False
            
            # Send email
            try:
                server.sendmail(smtp_config['smtp_username'], recipients, message.as_string())
                logger.info("Email sent successfully")
            except Exception as send_error:
                logger.error(f"Email sending failed: {send_error}")
                logger.error(traceback.format_exc())
                return False
            
            # Close connection
            server.quit()
            
            return True
        
        except Exception as connection_error:
            logger.error(f"SMTP Connection Error: {connection_error}")
            logger.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error in send_email: {e}")
        logger.error(traceback.format_exc())
        return False