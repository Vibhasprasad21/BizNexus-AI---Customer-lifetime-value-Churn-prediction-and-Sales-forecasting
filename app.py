import streamlit as st
import os
from streamlit.web.server.websocket_headers import _get_websocket_headers
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import SessionStateProxy



# Initialize session state for navigation
if "page_history" not in st.session_state:
    st.session_state.page_history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = None

def navigate_to(page_path, save_history=True):
    """
    Navigate to a specific page with improved error handling and history tracking.
    
    Args:
        page_path (str): Path to the page, e.g., "pages/01_Home.py" or "03_Upload"
        save_history (bool): Whether to save the current page in history for back navigation
    """
    # Normalize the path format
    if not page_path.startswith("pages/") and not page_path.endswith(".py"):
        # Convert format like "03_Upload" to "pages/03_Upload.py"
        page_path = f"pages/{page_path}.py"
    
    # Verify the page exists
    if not os.path.exists(page_path):
        st.error(f"Page '{page_path}' not found. Please check your application structure.")
        return False
    
    # Save current page to history if requested
    if save_history and st.session_state.current_page:
        st.session_state.page_history.append(st.session_state.current_page)
    
    # Update current page
    st.session_state.current_page = page_path
    
    # Navigate to the page
    try:
        st.switch_page(page_path)
        return True
    except Exception as e:
        st.error(f"Navigation error: {str(e)}")
        return False

def go_back():
    """Navigate to the previous page in history"""
    if st.session_state.page_history:
        previous_page = st.session_state.page_history.pop()
        # Don't save current page to history when going back
        navigate_to(previous_page, save_history=False)
    else:
        st.warning("No previous page in history.")

def get_current_user_info():
    """Get current user information from session state"""
    user_info = {
        "authenticated": st.session_state.get("authenticated", False),
        "user_info": st.session_state.get("user_info", None),
        "company_id": st.session_state.get("company_id", None)
    }
    return user_info

# Page configuration
st.set_page_config(
    page_title="BizNexus AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def check_for_sales_alerts():
    """Check for sales alerts if data is available"""
    if 'sales_data' in st.session_state:
        # Get logged-in user's email
        user_email = get_logged_in_user_email()
        
        if user_email:
            # Run analysis on sales data
            sales_data = st.session_state.sales_data
            processed_data, alerts = alert_system.analyze_sales_data(sales_data)
            
            # Send alert if issues detected
            if alerts:
                company_name = st.session_state.get('company_name', 'Your Business')
                alert_system.send_alert_email(alerts, user_email, company_name)


# Main app execution
def main():
    """Main application entry point"""
    # Check if the user is authenticated, otherwise redirect to authentication
    if not st.session_state.get("authenticated", False):
        # Redirect to authentication page if not already there
        current_script = get_script_run_ctx().main_script_path
        if not current_script.endswith("Home.py") and "authentication" not in current_script:
            navigate_to("pages/authentication.py")
    
    # Check for sales alerts after data upload
    if st.session_state.get("authenticated", False) and 'sales_data' in st.session_state:
        check_for_sales_alerts()
        
    ga_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    ga_property_id = os.environ.get('GA4_PROPERTY_ID')
    
    if not ga_credentials or not ga_property_id:
        if not current_script.endswith("Home.py") and not current_script.endswith("authentication.py"):
            st.warning("Google Analytics API is not fully configured. Some assistant features may be limited.")
    # If we're on the main app.py page, redirect to home
    current_script = get_script_run_ctx().main_script_path
    if current_script.endswith("app.py"):
        navigate_to("pages/01_Home.py")

if __name__ == "__main__":
    # Initialize session state if not already done
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None
    
    main()