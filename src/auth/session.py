import streamlit as st
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.source_util import get_pages

def requires_auth(page_func):
    """
    Decorator for Streamlit pages that require authentication.
    Redirects to authentication page if user is not logged in.
    
    Example usage:
    
    @requires_auth
    def main():
        st.title("Protected Page")
    
    if __name__ == "__main__":
        main()
    """
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            st.error("Please login to access this page")
            st.stop()
            # Optional: Redirect to login page
            nav_to("02_Authentication")
        else:
            # User is authenticated, execute the page function
            return page_func(*args, **kwargs)
    return wrapper

def nav_to(page_name: str):
    """Navigate to another page in the Streamlit app"""
    pages = get_pages("pages")
    for page_hash, page_config in pages.items():
        if page_config["page_name"] == page_name:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_config["page_name"],
                )
            )

def init_session_state():
    """Initialize session state variables if they don't exist"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "models" not in st.session_state:
        st.session_state.models = {
            "clv": None,
            "churn": None,
            "sales": None
        }
    if "notifications" not in st.session_state:
        st.session_state.notifications = []

def get_user_info():
    """
    Retrieve user information from Streamlit session state
    
    Returns:
        dict: User information or None if not available
    """
    if hasattr(st.session_state, 'user_info'):
        return st.session_state.user_info
    return None

def get_company_id():
    """Get the current user's company ID"""
    return st.session_state.get("company_id", None)

def get_user_full_name():
    """Get the current user's full name"""
    user_info = get_user_info()
    if user_info:
        return user_info.get("full_name", "User")
    return "User"

def get_company_name():
    """Get the current user's company name"""
    user_info = get_user_info()
    if user_info:
        return user_info.get("company_name", "Company")
    return "Company"

def is_authenticated():
    """Check if the user is authenticated"""
    return st.session_state.get("authenticated", False)

def logout():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.company_id = None
    # Optional: Clear other session state variables
    st.session_state.current_dataset = None
    st.session_state.analysis_complete = False
    st.session_state.models = {
        "clv": None,
        "churn": None,
        "sales": None
    }
    st.session_state.notifications = []
def check_login_status():
    """
    Check if user is logged in and return user information.
    
    Returns:
        tuple: (user_id, is_logged_in)
    """
    is_logged_in = st.session_state.get("authenticated", False)
    user_id = st.session_state.get("company_id") if is_logged_in else None
    
    return user_id, is_logged_in

def get_logged_in_user_email():
    """
    Retrieve the email of the logged-in user.
    
    Returns:
        str or None: Email of the logged-in user, or None if not logged in
    """
    if st.session_state.get("authenticated", False):
        user_info = st.session_state.get("user_info", {})
        return user_info.get("email")
    return None