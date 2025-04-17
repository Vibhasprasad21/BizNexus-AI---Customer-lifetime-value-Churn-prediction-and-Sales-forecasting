import pandas as pd
import numpy as np
from datetime import datetime
import re
from src.data_processing.upload import standardize_column_names, identify_column_types

def preprocess_dataset(df):
    """
    Preprocess the dataset using the comprehensive approach
    
    Args:
        df (pandas.DataFrame): The dataset to preprocess
        
    Returns:
        dict: Preprocessing result with the processed dataframe
    """
    return preprocess_data(df)

def preprocess_data(df):
    """
    Preprocess the input dataframe to prepare it for feature engineering
    
    Parameters:
    df (pandas.DataFrame): Raw input dataframe
    
    Returns:
    pandas.DataFrame: Preprocessed dataframe
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['Order Date', 'Ship Date', 'Signup_Date', 'First_Purchase_Date', 'Last_Purchase_Date', 
                   'purchase_date', 'transaction_date', 'order_date', 'ship_date']
    for col in date_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
    
    # Handle missing values
    # Fill numeric columns with appropriate values
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col in ['Sales', 'Profit', 'Discount', 'amount', 'revenue', 'sales']:
            # For financial metrics, 0 is often appropriate
            processed_df[col] = processed_df[col].fillna(0)
        else:
            # For other numeric columns, use median
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # Fill categorical columns with mode
    cat_cols = processed_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Unknown")
    
    # Fill date columns with appropriate values
    for col in date_columns:
        if col in processed_df.columns:
            # Forward fill for dates where possible
            processed_df[col] = processed_df[col].fillna(method='ffill')
            # For any remaining NaT values, use the minimum date
            if processed_df[col].isna().any() and not processed_df[col].dropna().empty:
                processed_df[col] = processed_df[col].fillna(processed_df[col].min())
    
    # Handle outliers in numeric columns using capping
    for col in ['Sales', 'Profit', 'Quantity', 'Discount', 'amount', 'revenue', 'sales']:
        if col in processed_df.columns:
            # Calculate Q1, Q3 and IQR
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers (1.5 * IQR)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap the outliers
            processed_df[col] = np.where(processed_df[col] < lower_bound, lower_bound, processed_df[col])
            processed_df[col] = np.where(processed_df[col] > upper_bound, upper_bound, processed_df[col])
    
    # Convert all string columns to lowercase
    for col in cat_cols:
        if processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].str.lower()
    
    # Ensure Customer ID is clean and consistent
    customer_id_columns = ['Customer ID', 'customer_id', 'CustomerID', 'Customer_ID']
    for col_name in customer_id_columns:
        if col_name in processed_df.columns:
            # Remove any special characters and whitespace
            processed_df[col_name] = processed_df[col_name].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)))
            # If this is not our primary customer_id column, rename it
            if col_name != 'customer_id':
                processed_df['customer_id'] = processed_df[col_name]
    
    # Ensure we have a customer_id column
    if 'customer_id' not in processed_df.columns and 'Customer ID' in processed_df.columns:
        processed_df['customer_id'] = processed_df['Customer ID']
    
    # Ensure Product ID is clean and consistent
    product_id_columns = ['Product ID', 'product_id', 'ProductID', 'Product_ID']
    for col_name in product_id_columns:
        if col_name in processed_df.columns:
            # Remove any special characters and whitespace
            processed_df[col_name] = processed_df[col_name].apply(lambda x: re.sub(r'[^a-zA-Z0-9-]', '', str(x)))
    
    # Create a clean copy of postal code if present
    postal_code_columns = ['Postal Code', 'postal_code', 'PostalCode', 'Postal_Code', 'ZIP', 'zip_code']
    for col_name in postal_code_columns:
        if col_name in processed_df.columns:
            processed_df['postal_code_clean'] = processed_df[col_name].astype(str).apply(lambda x: ''.join(filter(str.isdigit, x)))
    
    # Standardize column names for key fields
    # Ensure we have a purchase_date column
    if 'purchase_date' not in processed_df.columns:
        for col in ['Order Date', 'order_date', 'transaction_date']:
            if col in processed_df.columns:
                processed_df['purchase_date'] = processed_df[col]
                break
    
    # Ensure we have an amount column
    if 'amount' not in processed_df.columns:
        for col in ['Sales', 'sales', 'revenue', 'Revenue', 'total']:
            if col in processed_df.columns:
                processed_df['amount'] = processed_df[col]
                break
    
    # Drop duplicates if we have the necessary columns
    if 'Order ID' in processed_df.columns and 'Product ID' in processed_df.columns:
        processed_df.drop_duplicates(subset=['Order ID', 'Product ID'], keep='first', inplace=True)
    
    # Add a preprocessing timestamp
    processed_df['preprocessing_timestamp'] = datetime.now()
    
    return {
        'success': True,
        'dataframe': processed_df
    }

def aggregate_by_customer(df):
    """
    Aggregate the data by customer to prepare for customer-level analysis
    
    Parameters:
    df (pandas.DataFrame): Preprocessed dataframe
    
    Returns:
    pandas.DataFrame: Customer-level aggregated dataframe
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Map common column names to expected names
    column_mapping = {
        'customer_id': 'Customer ID',
        'purchase_date': 'Order Date',
        'amount': 'Sales',
        'profit': 'Profit',
        'quantity': 'Quantity',
        'order_id': 'Order ID',
        'product_id': 'Product ID'
    }
    
    # Apply the mapping for columns that exist
    for new_col, old_col in column_mapping.items():
        if new_col in df_copy.columns and old_col not in df_copy.columns:
            df_copy[old_col] = df_copy[new_col]
    
    # Ensure required columns exist
    required_cols = ['Customer ID', 'Order ID', 'Order Date', 'Sales']
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    # Add Profit column if not present (default to 20% of Sales)
    if 'Profit' not in df_copy.columns:
        df_copy['Profit'] = df_copy['Sales'] * 0.2
    
    # Add Quantity column if not present (default to 1)
    if 'Quantity' not in df_copy.columns:
        df_copy['Quantity'] = 1
    
    # Add Discount column if not present (default to 0)
    if 'Discount' not in df_copy.columns:
        df_copy['Discount'] = 0
    
    # Add Category column if not present (default to 'Unknown')
    if 'Category' not in df_copy.columns:
        df_copy['Category'] = 'Unknown'
    
    # Add Sub-Category column if not present (default to 'Unknown')
    if 'Sub-Category' not in df_copy.columns:
        df_copy['Sub-Category'] = 'Unknown'
    
    # Add returns and support columns if not present
    if 'Num_of_Returns' not in df_copy.columns:
        df_copy['Num_of_Returns'] = 0
    
    if 'Num_of_Support_Contacts' not in df_copy.columns:
        df_copy['Num_of_Support_Contacts'] = 0
    
    if 'Satisfaction_Score' not in df_copy.columns:
        df_copy['Satisfaction_Score'] = 0
    
    # Aggregations at the customer level
    customer_df = df_copy.groupby('Customer ID').agg({
        'Order ID': 'nunique',  # Number of unique orders
        'Order Date': ['min', 'max'],  # First and last order dates
        'Sales': 'sum',  # Total sales
        'Profit': 'sum',  # Total profit
        'Quantity': 'sum',  # Total quantity purchased
        'Discount': 'mean',  # Average discount
        'Product ID': 'nunique',  # Number of unique products purchased
        'Category': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",  # Most common category
        'Sub-Category': lambda x: x.mode()[0] if not x.mode().empty else "Unknown",  # Most common sub-category
        'Num_of_Returns': 'sum',  # Total returns if available
        'Num_of_Support_Contacts': 'sum',  # Total support contacts if available
        'Satisfaction_Score': 'mean',  # Average satisfaction score if available
    })
    
    # Flatten the multi-index columns
    customer_df.columns = ['_'.join(col).strip('_') for col in customer_df.columns.values]
    
    # Rename columns for clarity
    customer_df.rename(columns={
        'Order ID_nunique': 'Total_Orders',
        'Order Date_min': 'First_Order_Date',
        'Order Date_max': 'Last_Order_Date',
        'Sales_sum': 'Total_Sales',
        'Profit_sum': 'Total_Profit',
        'Quantity_sum': 'Total_Quantity',
        'Discount_mean': 'Avg_Discount',
        'Product ID_nunique': 'Unique_Products',
        'Category_<lambda>': 'Preferred_Category',
        'Sub-Category_<lambda>': 'Preferred_SubCategory',
        'Num_of_Returns_sum': 'Total_Returns',
        'Num_of_Support_Contacts_sum': 'Total_Support_Contacts',
        'Satisfaction_Score_mean': 'Avg_Satisfaction',
    }, inplace=True)
    
    # Calculate days since first and last purchase
    today = pd.Timestamp(datetime.now().date())
    customer_df['Days_Since_First_Purchase'] = (today - customer_df['First_Order_Date']).dt.days
    customer_df['Days_Since_Last_Purchase'] = (today - customer_df['Last_Order_Date']).dt.days
    
    # Calculate purchase frequency and average order value
    customer_df['Purchase_Frequency'] = customer_df['Total_Orders'] / customer_df['Days_Since_First_Purchase'] * 30  # Monthly frequency
    customer_df['Average_Order_Value'] = customer_df['Total_Sales'] / customer_df['Total_Orders']
    
    # Calculate customer value metrics
    customer_df['Customer_Value'] = customer_df['Total_Sales'] / customer_df['Days_Since_First_Purchase'] * 365  # Annual value
    customer_df['Customer_Margin'] = customer_df['Total_Profit'] / customer_df['Total_Sales'] * 100  # Profit margin percentage
    
    # Handle infinity and NaN values
    numeric_cols = customer_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        customer_df[col] = customer_df[col].replace([np.inf, -np.inf], np.nan)
        customer_df[col] = customer_df[col].fillna(0)
    
    # Create RFM scores (Recency, Frequency, Monetary)
    # Recency - lower is better
    customer_df['R_Score'] = pd.qcut(customer_df['Days_Since_Last_Purchase'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    # Frequency - higher is better
    customer_df['F_Score'] = pd.qcut(customer_df['Total_Orders'].clip(lower=1), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    # Monetary - higher is better
    customer_df['M_Score'] = pd.qcut(customer_df['Total_Sales'].clip(lower=0), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Convert scores to numeric if they're categorical
    for col in ['R_Score', 'F_Score', 'M_Score']:
        if customer_df[col].dtype.name == 'category':
            customer_df[col] = customer_df[col].astype(int)
    
    # Calculate RFM score
    customer_df['RFM_Score'] = customer_df['R_Score'] + customer_df['F_Score'] + customer_df['M_Score']
    
    # Create customer segments based on RFM score
    def segment_customer(row):
        score = row['RFM_Score']
        recency = row['R_Score']
        if score >= 13:
            return 'Champions'
        elif score >= 10:
            return 'Loyal Customers'
        elif score >= 9 and recency >= 4:
            return 'Potential Loyalists'
        elif score >= 9:
            return 'Recent Customers'
        elif 6 <= score <= 8:
            return 'Promising'
        elif 5 <= score <= 7 and recency <= 3:
            return 'Needs Attention'
        elif 4 <= score <= 5 and recency <= 2:
            return 'At Risk'
        elif 4 <= score <= 5:
            return 'Can\'t Lose'
        elif score <= 3 and recency <= 2:
            return 'Lost'
        else:
            return 'Hibernating'
    
    customer_df['Customer_Segment'] = customer_df.apply(segment_customer, axis=1)
    
    # Reset index to make Customer ID a column
    customer_df = customer_df.reset_index()
    
    # Merge with customer demographic info if available
    customer_info_cols = ['Customer ID', 'Customer Name', 'Segment', 'Age', 'Gender', 'Annual_Income', 
                          'Signup_Date', 'Years_as_Customer', 'Email_Opt_In', 'Marketing_Channel']
    
    # Filter only columns that exist in the dataframe
    available_customer_cols = [col for col in customer_info_cols if col in df_copy.columns]
    
    if available_customer_cols:
        # Get unique customer info (take the first occurrence for each customer)
        customer_info = df_copy.sort_values('Order Date').drop_duplicates(subset=['Customer ID'])[available_customer_cols]
        
        # Merge with the aggregated customer data
        customer_df = pd.merge(customer_df, customer_info, on='Customer ID', how='left')
    
    return customer_df

def engineer_features(df):
    """
    Engineer features for the three models (CLV, churn prediction, sales forecasting)
    
    Args:
        df (pandas.DataFrame): Preprocessed dataframe
        
    Returns:
        dict: Feature engineering results with dataframes for each model
    """
    # First try to get the customer-level aggregation
    try:
        customer_df = aggregate_by_customer(df)
        
        # Process for CLV model
        clv_df = customer_df.copy()
        
        # Process for churn model
        churn_df = customer_df.copy()
        
        # Add churn label based on recency (example: not purchased in 90 days)
        churn_df['Churned'] = (churn_df['Days_Since_Last_Purchase'] > 90).astype(int)
        
        # Process for sales forecasting
        # For sales forecasting, we need time series data
        # Let's create monthly aggregated sales
        if 'Order Date' in df.columns or 'purchase_date' in df.columns:
            date_col = 'Order Date' if 'Order Date' in df.columns else 'purchase_date'
            sales_col = 'Sales' if 'Sales' in df.columns else 'amount'
            
            df_copy = df.copy()
            df_copy['Year_Month'] = df_copy[date_col].dt.to_period('M')
            
            # Group by year-month and sum sales
            sales_forecast_df = df_copy.groupby('Year_Month')[sales_col].sum().reset_index()
            sales_forecast_df['Year_Month'] = sales_forecast_df['Year_Month'].astype(str)
        else:
            # If date column doesn't exist, create a dummy dataframe
            sales_forecast_df = pd.DataFrame({'Year_Month': [], 'Sales': []})
        
        return {
            'success': True,
            'customer_df': customer_df,
            'clv_df': clv_df,
            'churn_df': churn_df,
            'sales_forecast_df': sales_forecast_df
        }
    
    except Exception as e:
        # If the customer aggregation fails, fallback to a simpler approach
        return {
            'success': False,
            'error': str(e),
            'dataframe': df
        }