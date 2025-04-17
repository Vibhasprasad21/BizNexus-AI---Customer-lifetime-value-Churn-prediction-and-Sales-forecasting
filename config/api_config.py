import os
import streamlit as st
import streamlit as st
import google.generativeai as genai



def load_gemini_api_key():
    """
    Load Gemini API key securely 
    
    Returns:
        str: Gemini API key
    """
    # Try Streamlit secrets first
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass

    # Try environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        return api_key

    # Hardcoded fallback (REMOVE IN PRODUCTION!)
    # st.warning("Using a hardcoded API key - this is NOT secure!")
    # return "YOUR_ACTUAL_API_KEY"

    # If no API key found
    st.error("Gemini API key not found. Please set it in Streamlit secrets or environment variable.")
    return None

def configure_gemini_api():
    """
    Configure the Gemini API with the loaded API key
    
    Returns:
        bool: True if API is configured, False otherwise
    """
    api_key = load_gemini_api_key()
    if api_key:
        try:
            # Explicitly import and configure
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # List available models to debug
            models = genai.list_models()
            available_models = [model.name for model in models]
            st.info(f"Available models: {available_models}")
            
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}")
            return False
    return False
def setup_business_analytics_api():
    """Configure and check if Business Analytics API credentials are properly set up."""
    # Check if required environment variables are set
    has_credentials = "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
    has_property_id = "GA4_PROPERTY_ID" in os.environ
    has_bq_dataset = "BQ_DATASET_ID" in os.environ
    
    # Additional business-specific configuration check
    has_business_metrics = "BUSINESS_METRICS_ENABLED" in os.environ
    
    if not has_credentials:
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    if not has_property_id:
        print("Warning: GA4_PROPERTY_ID environment variable not set.")
    if not has_bq_dataset:
        print("Warning: BQ_DATASET_ID environment variable not set. BigQuery data access will be limited.")
    if not has_business_metrics:
        print("Warning: BUSINESS_METRICS_ENABLED not set. Setting to default 'true'.")
        os.environ["BUSINESS_METRICS_ENABLED"] = "true"
    
    # Return overall status
    api_configured = has_credentials and has_property_id
    
    if api_configured:
        print("✅ Business Analytics API successfully configured!")
    else:
        print("⚠️ Business Analytics API configuration incomplete. Some features will be limited.")
    
    return api_configured

def get_available_business_metrics():
    """Return a list of available business metrics that can be queried."""
    # Core business metrics for analysis
    core_metrics = [
        "revenue", "averagePurchaseValue", "purchaseToViewRate",
        "returnOnAdSpend", "transactionsPerUser", "customerAcquisitionCost",
        "cartToDetailRate", "conversionRate", "totalRevenue"
    ]
    
    # Advanced metrics (only available with BUSINESS_METRICS_ENABLED)
    if os.environ.get("BUSINESS_METRICS_ENABLED", "false").lower() == "true":
        advanced_metrics = [
            "customerLifetimeValue", "churnRate", "customerRetentionRate",
            "averageOrderValue", "repeatPurchaseRate", "profitMargin",
            "salesGrowthRate", "revenueByChannel", "productPerformance"
        ]
        return core_metrics + advanced_metrics
    
    return core_metrics