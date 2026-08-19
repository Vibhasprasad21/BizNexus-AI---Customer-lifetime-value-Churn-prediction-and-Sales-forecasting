import logging

import streamlit as st

from src.database.store import save_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_alert_email(to_email, subject, message, from_email=None):
    """
    Record an in-app alert notification (formerly sent as an email).

    Kept the original signature so existing call sites don't need changes -
    only the delivery mechanism changed, from SMTP to a local SQLite record
    shown in the app's notifications panel.

    Args:
        to_email (str or list): Recipient email address(es) - kept for
            signature compatibility, not used for delivery anymore.
        subject (str): Notification subject
        message (str): Notification body
        from_email (str, optional): Unused, kept for signature compatibility.

    Returns:
        bool: True if the notification was recorded, False otherwise
    """
    try:
        company_id = st.session_state.get('company_id')
        if not company_id:
            logger.error("No company_id in session state; cannot record notification")
            return False

        full_message = f"{subject}\n\n{message}" if subject else message

        result = save_notification(
            company_id=company_id,
            notification_type='alert',
            message=full_message,
            priority='high'
        )

        return result.get('success', False)

    except Exception as e:
        logger.error(f"Error recording in-app alert: {e}")
        return False
