import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import tempfile

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location and return the path
    
    Args:
        uploaded_file (UploadedFile): The uploaded file from Streamlit
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        return tmp_path
    except Exception as e:
        st.error(f"Error saving uploaded file: {str(e)}")
        return None

def validate_dataset(df):
    """
    Validate if the uploaded dataset has the required columns and structure
    
    Args:
        df (pandas.DataFrame): The uploaded dataset
        
    Returns:
        dict: Validation result with status and message
    """
    issues = []
    
    # Create a lowercase version of column names for case-insensitive comparison
    columns_lower = [col.lower() for col in df.columns]
    
    # Check if dataframe is empty
    if df.empty:
        return {
            'valid': False,
            'message': 'The uploaded file is empty. Please upload a file with data.'
        }
    
    # Check for required customer ID column - case insensitive check
    customer_id_variants = ['customer_id', 'customer id', 'customerid', 'cust_id', 'custid']
    has_customer_id = any(variant in columns_lower or variant.replace('_', ' ') in columns_lower for variant in customer_id_variants)
    
    if not has_customer_id and 'customer id' not in ' '.join(columns_lower):
        issues.append("Missing required Customer ID column. Please ensure your data has a customer identifier column.")
    
    # Check for date column - case insensitive check
    date_column_variants = ['purchase_date', 'purchase date', 'order_date', 'order date', 
                          'transaction_date', 'transaction date', 'date']
    has_date_column = any(variant in columns_lower or variant.replace('_', ' ') in columns_lower for variant in date_column_variants)
    
    if not has_date_column:
        issues.append("Missing date column: The dataset should have Order Date, Purchase Date, or Transaction Date.")
    
    # Check for sales/amount column - case insensitive check
    amount_column_variants = ['amount', 'sales', 'revenue', 'price', 'total']
    has_amount_column = any(variant in columns_lower for variant in amount_column_variants)
    
    if not has_amount_column and 'sales' not in columns_lower:
        issues.append("Missing amount column: The dataset should have Sales, Amount, Revenue, or similar.")
    
    # Check for any completely empty columns
    empty_columns = [col for col in df.columns if df[col].isna().all()]
    if empty_columns:
        issues.append(f"The following columns are completely empty: {', '.join(empty_columns)}")
    
    # Check for any columns with all identical values
    identical_columns = [col for col in df.columns if df[col].nunique() == 1]
    if identical_columns:
        issues.append(f"The following columns have all identical values: {', '.join(identical_columns)}")
    
    # Validate the dataset is valid if no critical issues
    # We'll still allow the dataset if there are only warnings about empty/identical columns
    critical_issues = [issue for issue in issues if 
                      "Missing required" in issue or 
                      "Missing date column" in issue or 
                      "Missing amount column" in issue]
    
    is_valid = len(critical_issues) == 0
    
    result = {
        'valid': is_valid,
        'message': 'Dataset is valid' if is_valid else 'Dataset has validation issues',
        'issues': issues
    }
    
    return result

def show_dataset_preview(df):
    """
    Display a preview of the dataset with stats
    
    Args:
        df (pandas.DataFrame): The dataset to preview
    """
    # Show the first 5 rows
    st.markdown("<h4>Data Preview</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(5), use_container_width=True)
    
    # Dataset stats
    st.markdown("<h4>Dataset Statistics</h4>", unsafe_allow_html=True)
    
    # Basic stats in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", format(df.shape[0], ","))
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        # Try different variants of customer ID column
        for col_name in ['Customer ID', 'customer_id', 'CustomerID', 'Customer_ID']:
            if col_name in df.columns:
                st.metric("Unique Customers", format(df[col_name].nunique(), ","))
                break
        else:
            st.metric("Unique Customers", "N/A")
    with col4:
        missing_percent = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("Missing Values", f"{missing_percent:.1f}%")
    
    # Column information
    st.markdown("<h4>Column Information</h4>", unsafe_allow_html=True)
    
    # Create a dataframe with column info
    column_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isna().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Sample Values': [str(df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()) if not df[col].dropna().empty else "[]" for col in df.columns]
    })
    
    st.dataframe(column_info, use_container_width=True)

def identify_column_types(df):
    """
    Identify the type of each column in the dataset (categorical, numerical, date, ID)
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        dict: Column types mapping
    """
    column_types = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for IDs
        if 'id' in col_lower or col_lower.endswith('_id') or col_lower.startswith('id_'):
            column_types[col] = 'id'
            continue
        
        # Check for dates
        if 'date' in col_lower or 'time' in col_lower or 'day' in col_lower or 'month' in col_lower or 'year' in col_lower:
            column_types[col] = 'date'
            continue
        
        # Check data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's a categorical variable encoded as numeric
            if df[col].nunique() < 10 and df[col].nunique() / len(df[col]) < 0.05:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'numerical'
        elif pd.api.types.is_string_dtype(df[col]):
            # Check if it's actually a date string
            date_pattern = re.compile(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}')
            sample = df[col].dropna().sample(min(100, len(df[col].dropna())))
            date_matches = sum(1 for val in sample if date_pattern.search(str(val)))
            
            if date_matches > len(sample) * 0.8:
                column_types[col] = 'date'
            elif df[col].nunique() < 10 or df[col].nunique() / len(df[col]) < 0.05:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
        else:
            column_types[col] = 'other'
    
    return column_types

def standardize_column_names(df):
    """
    Standardize column names in the dataset
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        pandas.DataFrame: Dataset with standardized column names
    """
    # Make a copy of the dataframe
    standardized_df = df.copy()
    
    # Mapping of common column name variations
    column_mapping = {
        # ID columns
        'customer id': 'customer_id',
        'customerid': 'customer_id',
        'cust_id': 'customer_id',
        'cust id': 'customer_id',
        'client id': 'customer_id',
        'clientid': 'customer_id',
        'client_id': 'customer_id',
        'user id': 'customer_id',
        'userid': 'customer_id',
        'user_id': 'customer_id',
        
        # Date columns
        'transaction date': 'purchase_date',
        'transaction_date': 'purchase_date',
        'order date': 'purchase_date',
        'order_date': 'purchase_date',
        'date': 'purchase_date',
        'purchase date': 'purchase_date',
        
        # Amount columns
        'transaction amount': 'amount',
        'transaction_amount': 'amount',
        'order amount': 'amount',
        'order_amount': 'amount',
        'revenue': 'amount',
        'sales': 'amount',
        'price': 'amount',
        'total': 'amount',
        'total_amount': 'amount',
        'total amount': 'amount',
        
        # Customer columns
        'customer name': 'customer_name',
        'customername': 'customer_name',
        'cust_name': 'customer_name',
        'cust name': 'customer_name',
        'client name': 'customer_name',
        'clientname': 'customer_name',
        'client_name': 'customer_name',
        
        # Other common columns
        'product id': 'product_id',
        'productid': 'product_id',
        'product_id': 'product_id',
        'product name': 'product_name',
        'productname': 'product_name',
        'product_name': 'product_name',
        'quantity': 'quantity',
        'qty': 'quantity',
        'category': 'category',
        'product category': 'category',
        'product_category': 'category',
        'region': 'region',
        'city': 'city',
        'state': 'state',
        'country': 'country',
        'zip': 'postal_code',
        'zipcode': 'postal_code',
        'zip_code': 'postal_code',
        'postal code': 'postal_code',
        'postalcode': 'postal_code',
        'gender': 'gender',
        'age': 'age',
        'email': 'email',
        'phone': 'phone',
        'address': 'address'
    }
    
    # Standardize column names
    standardized_df.columns = [
        column_mapping.get(col.lower(), col) for col in standardized_df.columns
    ]
    
    return standardized_df