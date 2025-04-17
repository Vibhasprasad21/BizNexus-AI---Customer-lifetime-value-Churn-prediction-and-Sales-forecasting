import streamlit as st
from firebase_admin import firestore
import pandas as pd
from datetime import datetime
import traceback

def main():
    st.title("Alert Settings")
    
    if not st.session_state.authenticated:
        st.warning("Please log in to access this page.")
        return
    
    company_id = st.session_state.company_id
    
    # Get Firestore instance
    db = firestore.client()
    
    # Check if system-wide SMTP settings exist in secrets
    system_smtp_config = hasattr(st, 'secrets') and 'email' in st.secrets

    # Create tabs for different alert types
    tab1, tab2 = st.tabs(["Churn Risk Alerts", "Sales Risk Alerts"])
    
    # Rest of the code remains the same as in the original file
    # ... (previous code)

    # SMTP Configuration section
    st.header("Email Configuration")
    
    # First, inform about system-wide SMTP if it exists
    if system_smtp_config:
        st.info("🌐 System-wide SMTP configuration is available and will be used by default.")
        st.warning("If you want to override, you can still configure user-specific settings below.")
    
    # Get current SMTP configuration
    user_id = st.session_state.user_info.get('user_id') if st.session_state.user_info else None
    
    if user_id:
        user_doc = db.collection('users').document(user_id).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            email_config = user_data.get('email_config', {})
            
            with st.form("smtp_settings"):
                st.subheader("SMTP Server Settings")
                
                # Add clear instructions about system-wide config
                if system_smtp_config:
                    st.info("💡 You can override system-wide settings here or leave blank to use system defaults.")
                else:
                    st.info("These settings are required to send email alerts. For Gmail, you'll need to use an App Password.")
                
                smtp_host = st.text_input(
                    "SMTP Server",
                    value=email_config.get('smtp_host', ''),
                    placeholder='Leave blank to use system default (smtp.gmail.com)'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    smtp_port = st.number_input(
                        "SMTP Port",
                        min_value=1,
                        max_value=65535,
                        value=email_config.get('smtp_port', 587),
                        help="Default is 587 for most email providers"
                    )
                
                with col2:
                    use_tls = st.checkbox(
                        "Use TLS",
                        value=email_config.get('use_tls', True)
                    )
                
                smtp_username = st.text_input(
                    "SMTP Username (Email Address)",
                    value=email_config.get('smtp_username', ''),
                    placeholder='Leave blank to use system default'
                )
                
                smtp_password = st.text_input(
                    "SMTP Password (App Password for Gmail)",
                    type="password",
                    value="●●●●●●●●●●●●" if email_config.get('smtp_password') else "",
                    help="Leave blank to use system default password"
                )
                
                submit = st.form_submit_button("Save SMTP Settings")
            
            if submit:
                # Prepare configuration to save
                config_to_save = {
                    'smtp_host': smtp_host or None,
                    'smtp_port': smtp_port,
                    'use_tls': use_tls,
                    'smtp_username': smtp_username or None,
                    'smtp_password': smtp_password if smtp_password != "●●●●●●●●●●●●" else None,
                }
                
                # Only save non-None values
                config_to_save = {k: v for k, v in config_to_save.items() if v is not None}
                
                # Add configured flag if we have enough info
                if config_to_save.get('smtp_username'):
                    config_to_save['configured'] = True
                
                # Update user document
                db.collection('users').document(user_id).update({
                    'email_config': config_to_save
                })
                
                st.success("SMTP settings saved successfully!")
                
                # Test email option
                if st.button("Send Test Email"):
                    from src.utils.email_service import send_email
                    import logging
                    import traceback
                    
                    # Use the username for sending (either from saved config or system default)
                    recipient = smtp_username or (st.secrets.email.get('smtp_username') if system_smtp_config else None)
                    
                    if not recipient:
                        st.error("No email address available to send test email.")
                        return
                    
                    subject = "BizNexus AI - Test Email"
                    html_content = f"""
                    <html>
                    <body>
                        <h2>BizNexus AI Test Email</h2>
                        <p>This is a test email to verify your SMTP settings are working correctly.</p>
                        <p>If you received this email, your alert system is configured properly!</p>
                        <p>Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </body>
                    </html>
                    """
                    
                    try:
                        # Set up detailed logging
                        logging.basicConfig(level=logging.INFO)
                        logger = logging.getLogger(__name__)
                        
                        # Attempt to send email using user_id to pick up custom or system config
                        success = send_email([recipient], subject, html_content, user_id)
                        
                        if success:
                            st.success(f"✅ Test email sent successfully to {recipient}!")
                        else:
                            st.error("❌ Failed to send test email. Please review your SMTP settings.")
                            st.warning("Troubleshooting tips:")
                            st.warning("1. Verify SMTP Username and Password")
                            st.warning("2. Check Internet Connection")
                            st.warning("3. Ensure Email Provider Allows SMTP")
                    
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.error(traceback.format_exc())

if __name__ == "__main__":
    # Initialize session state if not already done
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None
        
    main()