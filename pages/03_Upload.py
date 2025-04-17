import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import os
from datetime import datetime

# Import necessary modules
from src.auth.session import requires_auth, get_user_info, get_company_id, nav_to
from src.firebase.firestore import save_dataset
from src.data_processing.upload import validate_dataset, show_dataset_preview, save_uploaded_file
from src.data_processing.preprocessing import preprocess_dataset
from src.data_processing.feature_engineering import engineer_features
import streamlit.components.v1 as components

# Create downloads directory
DOWNLOADS_DIR = os.path.join(os.getcwd(), 'downloads', 'processed_datasets')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="BizNexus AI | Upload Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;  /* Soft pastel blue-gray background */
            color: #2c3e50;  /* Dark text for readability */
        }
        
        .upload-container {
            background-color: #ffffff;  /* Clean white background */
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        }
        
        .stButton > button {
            background-color: #6a89cc !important;  /* Soft pastel blue */
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #5a7baf !important;  /* Slightly darker shade */
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        .upload-area {
            border: 2px dashed #6a89cc;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #f0f4f8;  /* Light pastel background */
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #5a7baf;
            background-color: #e6eaf4;  /* Slightly different pastel shade */
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """
    Render the header with logo and user welcome
    """
    # Get user information
    user_info = get_user_info()
    user_name = user_info.get('full_name', 'User') if user_info else 'User'
    company_name = user_info.get('company_name', 'Company') if user_info else 'Company'
    
    # Create columns for logo and welcome message
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display BizNexus AI logo
        logo_path = "assets/images/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
        else:
            st.warning("Logo not found")
    
    with col2:
        # Welcome message
        st.markdown(f"""
        <div style="background-color:#f0f4f8; padding:20px; border-radius:10px;">
            <h2 style="margin-bottom:10px; color:#2c3e50;">Welcome, {user_name}!</h2>
            <p style="color:#6a89cc;">Company: {company_name}</p>
            <p style="color:#7f8c8d; font-size:0.9em;">
                Ready to unlock insights from your business data? 
                Upload your dataset to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

@requires_auth
def main():
    # Load CSS 
    load_css()
    
    # Render header with logo and welcome message
    render_header()
    
    # Title
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px; color: #2c3e50;'>Upload Your Business Data</h1>", unsafe_allow_html=True)
    
    # Main container
    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        
        # Upload instructions
        st.markdown("""
        <div class="info-box" style="background-color: #f0f4f8; padding: 15px; border-radius: 8px;">
            <h3 style="color: #6a89cc;">📋 Data Upload Instructions</h3>
            <p>Upload your customer data CSV or Excel file to unleash advanced AI analytics.</p>
            <p>Required Columns:</p>
            <ul>
                <li><strong>Customer ID</strong>: Unique customer identifier</li>
                <li><strong>Order Date</strong>: Purchase date (YYYY-MM-DD)</li>
                <li><strong>Sales/Amount</strong>: Purchase amount</li>
                <li><strong>Product ID</strong>: Product identifier</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload area
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon" style="color: #6a89cc; font-size: 48px;">📤</div>
            <h3 style="color: #5a7baf;">Drag and Drop or Select Your File</h3>
            <p>Supports CSV and Excel files (XLS, XLSX)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Actual file uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Upload Successful Message
            st.success("File uploaded successfully! 🎉")
            
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Reset the file pointer to the beginning
                uploaded_file.seek(0)
                
                # Basic validation
                validation_result = validate_dataset(df)
                
                if validation_result['valid']:
                    # Show preview
                    show_dataset_preview(df)
                    
                    # Dataset metadata form
                    st.markdown("<h3>Dataset Information</h3>", unsafe_allow_html=True)
                    
                    # Two columns for form inputs
                    col1, col2 = st.columns(2)
                    with col1:
                        dataset_name = st.text_input("Dataset Name", value=f"{uploaded_file.name.split('.')[0]}")
                    
                    with col2:
                        dataset_description = st.text_input("Description (optional)", placeholder="Brief dataset description")
                    
                    # Preprocessing
                    with st.spinner("Preprocessing data..."):
                        preprocessing_result = preprocess_dataset(df)
                        processed_df = preprocessing_result.get('dataframe', df)
                    
                    # Feature Engineering - Without spinner to avoid continued display
                    try:
                        # Perform feature engineering
                        result_df, engineered_features = engineer_features(processed_df)
                        
                        # Get customer and sales time series dataframes
                        customer_df = result_df.attrs.get('customer_df', pd.DataFrame())
                        sales_ts_df = result_df.attrs.get('sales_ts_df', pd.DataFrame())
                        
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        local_filename = f"{dataset_name}_engineered_{timestamp}.csv"
                        local_filepath = os.path.join(DOWNLOADS_DIR, local_filename)
                        
                        # Save engineered features locally
                        result_df.to_csv(local_filepath, index=False)
                        
                        # Prompt to download engineered features
                        st.markdown("### 📊 Engineered Features Dataset")
                        with open(local_filepath, 'rb') as file:
                            download_button = st.download_button(
                                label="Download Engineered Features CSV",
                                data=file,
                                file_name=local_filename,
                                mime='text/csv',
                                help="Download the dataset with engineered features"
                            )
                        
                        # Store in session state
                        st.session_state.result_df = result_df
                        st.session_state.customer_df = customer_df
                        st.session_state.sales_ts_df = sales_ts_df
                        st.session_state.engineered_features = engineered_features
                        st.session_state.features_engineered = True
                        
                        # Process and Analyze Button
                        if st.button("Process & Analyze Data", type="primary", use_container_width=True):
                            # Use direct navigation like in the authentication file
                            try:
                                st.switch_page("pages/04_CLV_Analysis.py")
                            except Exception as e:
                                st.error(f"Navigation error: {str(e)}")
                                st.info("Please manually navigate to the CLV Analysis page.")
                        
                    except Exception as e:
                        st.error(f"Feature Engineering Error: {str(e)}")
                        st.exception(e)
                
                else:
                    st.error(validation_result['message'])
                    
                    # Show the list of specific issues
                    if 'issues' in validation_result and validation_result['issues']:
                        st.markdown("<h4>The following issues were found:</h4>", unsafe_allow_html=True)
                        for issue in validation_result['issues']:
                            st.markdown(f"- {issue}")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the page
if __name__ == "__main__":
    main()