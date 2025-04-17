import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.data_processing.preprocessing import aggregate_by_customer


def engineer_features(df):
    """
    Engineer features for customer analytics, CLV calculation, churn prediction, 
    and sales forecasting
    
    Parameters:
    df (pandas.DataFrame): Preprocessed dataframe
    
    Returns:
    pandas.DataFrame: Dataframe with engineered features
    list: List of engineered feature names
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['Order Date', 'Ship Date', 'Signup_Date', 'First_Purchase_Date', 'Last_Purchase_Date']
    for col in date_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
    
    # Ensure Customer ID and other essential columns are present
    essential_columns = ['Customer ID', 'Customer Name', 'Order ID', 'Sales']
    for col in essential_columns:
        if col not in processed_df.columns:
            raise ValueError(f"Required column {col} is missing from the dataset")
    
    # Aggregate by customer for customer-level analysis
    customer_df = aggregate_by_customer(processed_df)
    
    # List to track engineered features
    engineered_features = []
    
    # Add current date for time-based calculations
    current_date = processed_df['Order Date'].max() + timedelta(days=1)
    
    # If 'current_date' is NaT, use the current datetime
    if pd.isna(current_date):
        current_date = datetime.now()
    
    # ======== Time-based Metrics ========
    
    # 1. Recency - days since last purchase
    customer_df['Recency'] = (current_date - customer_df['Last_Order_Date']).dt.days
    engineered_features.append('Recency')
    
    # 2. Frequency - number of orders per time period
    customer_df['Frequency'] = customer_df['Total_Orders']
    engineered_features.append('Frequency')
    
    # Calculate customer tenure in days
    customer_df['Tenure_Days'] = (customer_df['Last_Order_Date'] - customer_df['First_Order_Date']).dt.days
    customer_df['Tenure_Days'] = customer_df['Tenure_Days'].apply(lambda x: max(x, 1))  # Avoid division by zero
    engineered_features.append('Tenure_Days')
    
    # 3. Purchase Frequency (orders per day)
    customer_df['Purchase_Frequency'] = customer_df['Total_Orders'] / customer_df['Tenure_Days']
    engineered_features.append('Purchase_Frequency')
    
    # 4. Average time between purchases
    customer_df['Avg_Time_Between_Purchases'] = customer_df['Tenure_Days'] / customer_df['Total_Orders']
    engineered_features.append('Avg_Time_Between_Purchases')
    
    # 5. Purchase Seasonality (requires order-level data analysis)
    # Group orders by month and calculate monthly purchase frequencies
    order_months = processed_df.groupby(['Customer ID', processed_df['Order Date'].dt.month]).size().unstack(fill_value=0)
    
    # Calculate coefficient of variation across months to measure seasonality
    if not order_months.empty:
        order_months_cv = order_months.apply(lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0, axis=1)
        seasonality_df = pd.DataFrame(order_months_cv, columns=['Purchase_Seasonality'])
        seasonality_df.reset_index(inplace=True)
        
        # Merge seasonality metric with customer dataframe
        customer_df = pd.merge(customer_df, seasonality_df, on='Customer ID', how='left')
        customer_df['Purchase_Seasonality'].fillna(0, inplace=True)
        engineered_features.append('Purchase_Seasonality')
    
    # ======== Financial Metrics ========
    
    # 1. Average Transaction Amount
    customer_df['Average_Transaction_Amount'] = customer_df['Total_Sales'] / customer_df['Total_Orders']
    engineered_features.append('Average_Transaction_Amount')
    
    # 2. Monetary Value (Average order value)
    customer_df['Monetary_Value'] = customer_df['Total_Sales'] / customer_df['Total_Orders']
    engineered_features.append('Monetary_Value')
    
    # 3. Annualized Spend
    # Calculate tenure in years
    customer_df['Tenure_Years'] = customer_df['Tenure_Days'] / 365
    customer_df['Tenure_Years'] = customer_df['Tenure_Years'].apply(lambda x: max(x, 0.1))  # Minimum 0.1 years to avoid division issues
    
    customer_df['Annualized_Spend'] = customer_df['Total_Sales'] / customer_df['Tenure_Years']
    engineered_features.append('Annualized_Spend')
    
    # ======== Behavioral Indicators ========
    
   
    

    
  
    
    # 4. Promotion Responsiveness
    if 'Promotion_Response' in processed_df.columns:
        # Aggregate promotion response by customer
        promo_df = processed_df.groupby('Customer ID')['Promotion_Response'].mean().reset_index()
        
        # Merge with customer dataframe
        customer_df = pd.merge(customer_df, promo_df, on='Customer ID', how='left')
        customer_df['Promotion_Responsiveness'] = customer_df['Promotion_Response']
        customer_df['Promotion_Responsiveness'].fillna(0, inplace=True)
        engineered_features.append('Promotion_Responsiveness')
    
  
    
 

    
    # 8. Reduced Purchase Activity
    # Compare recent purchase frequency with overall
    
    # Create a function to calculate recent activity change
    def calc_recent_activity(customer_id):
        # Get customer orders
        customer_orders = processed_df[processed_df['Customer ID'] == customer_id].sort_values('Order Date')
        
        if len(customer_orders) <= 1:
            return 0  # Not enough orders to calculate trend
        
        # Get first and last order dates
        first_date = customer_orders['Order Date'].min()
        last_date = customer_orders['Order Date'].max()
        
        # Calculate the total tenure
        total_tenure = (last_date - first_date).days
        if total_tenure <= 0:
            return 0
        
        # Define the "recent" period as the last 30% of the tenure
        recent_start = last_date - timedelta(days=0.3 * total_tenure)
        
        # Count orders in recent period
        total_orders = len(customer_orders)
        recent_orders = len(customer_orders[customer_orders['Order Date'] >= recent_start])
        
        # Calculate the expected recent orders (proportional to time)
        expected_recent = total_orders * 0.3
        
        # Return the difference (negative means reduced activity)
        return (recent_orders - expected_recent) / expected_recent if expected_recent > 0 else 0
    
    # Apply the function to calculate reduced purchase activity
    # This can be computationally intensive for large datasets
    if len(processed_df['Customer ID'].unique()) > 1000:
        # Sample customers for larger datasets
        sample_customers = np.random.choice(
            processed_df['Customer ID'].unique(),
            size=min(1000, len(processed_df['Customer ID'].unique())),
            replace=False
        )
        activity_dict = {cust_id: calc_recent_activity(cust_id) for cust_id in sample_customers}
        
        # Create DataFrame and merge
        activity_df = pd.DataFrame(list(activity_dict.items()), columns=['Customer ID', 'Recent_Activity_Change'])
        customer_df = pd.merge(customer_df, activity_df, on='Customer ID', how='left')
        customer_df['Recent_Activity_Change'].fillna(0, inplace=True)
    else:
        # Process all customers for smaller datasets
        customer_df['Recent_Activity_Change'] = customer_df['Customer ID'].apply(calc_recent_activity)
    
    # Convert to a flag for reduced activity
    customer_df['Reduced_Purchase_Activity'] = (customer_df['Recent_Activity_Change'] < -0.1).astype(int)
    engineered_features.append('Reduced_Purchase_Activity')
    
    # ======== Segment Analysis ========
    
    # 1. RFM Segmentation
    # Standardize RFM metrics
    rfm_features = ['Recency', 'Frequency', 'Monetary_Value']
    rfm_data = customer_df[rfm_features].copy()
    
    # Invert recency (lower is better)
    rfm_data['Recency'] = rfm_data['Recency'].max() - rfm_data['Recency']
    
    # Standardize the features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_df['RFM_Segment'] = kmeans.fit_predict(rfm_scaled)
    engineered_features.append('RFM_Segment')
    
    # 2. High Spender Flag (top 20%)
    sales_threshold = customer_df['Total_Sales'].quantile(0.8)
    customer_df['High_Spender_Flag'] = (customer_df['Total_Sales'] >= sales_threshold).astype(int)
    engineered_features.append('High_Spender_Flag')
    
    # 3. Region-Based Spend (if region data available)
    if 'Region' in processed_df.columns:
        # Calculate average spend by region
        region_avg_spend = processed_df.groupby('Region')['Sales'].mean().reset_index()
        region_avg_spend.columns = ['Region', 'Region_Avg_Spend']
        
        # Get customer's primary region
        customer_regions = processed_df.groupby(['Customer ID', 'Region']).size().reset_index(name='count')
        customer_regions = customer_regions.sort_values(['Customer ID', 'count'], ascending=[True, False])
        customer_primary_region = customer_regions.drop_duplicates('Customer ID')[['Customer ID', 'Region']]
        
        # Merge with customer data
        customer_df = pd.merge(customer_df, customer_primary_region, on='Customer ID', how='left')
        
        # Add region average spend
        region_mapping = dict(zip(region_avg_spend['Region'], region_avg_spend['Region_Avg_Spend']))
        customer_df['Region_Avg_Spend'] = customer_df['Region'].map(region_mapping)
        customer_df['Region_Avg_Spend'].fillna(customer_df['Region_Avg_Spend'].mean(), inplace=True)
        
        # Calculate relative spend (compared to region average)
        customer_df['Region_Relative_Spend'] = customer_df['Average_Transaction_Amount'] / customer_df['Region_Avg_Spend']
        engineered_features.append('Region_Relative_Spend')
    
    # 4. Top Product Category
    if 'Category' in processed_df.columns:
        # Calculate the most purchased category by each customer
        category_counts = processed_df.groupby(['Customer ID', 'Category']).size().reset_index(name='category_count')
        top_categories = category_counts.sort_values(['Customer ID', 'category_count'], ascending=[True, False])
        top_categories = top_categories.drop_duplicates('Customer ID')[['Customer ID', 'Category']]
        top_categories.columns = ['Customer ID', 'Top_Product_Category']
        
        # Merge with customer dataframe
        customer_df = pd.merge(customer_df, top_categories, on='Customer ID', how='left')
        engineered_features.append('Top_Product_Category')
    
    # ======== Prediction Targets ========
    
    # 1. Customer Lifetime Value (CLV) Calculation
    
    # Simple CLV calculation
    # CLV = Average Order Value × Purchase Frequency × Customer Lifespan
    # Estimate customer lifespan as 3x the current tenure
    customer_df['Estimated_Lifespan'] = customer_df['Tenure_Days'] * 3
    customer_df['Estimated_Lifespan'] = customer_df['Estimated_Lifespan'].clip(lower=365)  # Minimum 1 year
    
    customer_df['CLV'] = customer_df['Average_Transaction_Amount'] * customer_df['Purchase_Frequency'] * customer_df['Estimated_Lifespan']
    
    # Add discounted CLV calculation with 10% annual discount rate
    discount_rate = 0.1
    daily_discount_rate = (1 + discount_rate)**(1/365) - 1
    
    # Simplify by using continuous discounting formula
    customer_df['Discounted_CLV'] = customer_df['Average_Transaction_Amount'] * customer_df['Purchase_Frequency'] * \
                                  (1 - np.exp(-customer_df['Estimated_Lifespan'] * np.log(1 + daily_discount_rate))) / \
                                  np.log(1 + daily_discount_rate)
    
    # Clip CLV to reasonable values
    customer_df['CLV'] = customer_df['CLV'].clip(lower=0)
    customer_df['Discounted_CLV'] = customer_df['Discounted_CLV'].clip(lower=0)
    engineered_features.extend(['CLV', 'Discounted_CLV'])
    
    # 2. Churn Prediction Label
    # Define churn based on recency compared to the customer's average purchase interval
    
    # Create churn label based on recency compared to average time between purchases
    customer_df['Expected_Purchase_Gap'] = customer_df['Avg_Time_Between_Purchases'].clip(lower=1, upper=365)
    customer_df['Churn_Threshold_Days'] = customer_df['Expected_Purchase_Gap'] * 3  # 3x the average gap
    
    # Label customers as churned if their recency exceeds the threshold
    customer_df['Churn_Label'] = (customer_df['Recency'] > customer_df['Churn_Threshold_Days']).astype(int)
    engineered_features.append('Churn_Label')
    
    # Also create timeframe-specific churn predictions
    for timeframe_days in [30, 60, 90]:
        future_date = current_date + timedelta(days=timeframe_days)
        
        # Predict churn by extrapolating from current behavior
        # If the next predicted purchase would occur after the timeframe, label as likely to churn
        customer_df['Next_Predicted_Purchase'] = customer_df['Last_Order_Date'] + pd.to_timedelta(customer_df['Expected_Purchase_Gap'] * 1.5, unit='d')
        customer_df[f'Churn_Prediction_{timeframe_days}d'] = (customer_df['Next_Predicted_Purchase'] > future_date).astype(int)
        engineered_features.append(f'Churn_Prediction_{timeframe_days}d')
    
    # 3. Customer Value Tier
    # Create customer value tiers based on CLV quartiles
    customer_df['CLV_Quartile'] = pd.qcut(customer_df['CLV'], 4, labels=[0, 1, 2, 3])
    customer_df['Value_Tier'] = customer_df['CLV_Quartile'].map({0: 'Low', 1: 'Medium', 2: 'High', 3: 'Premium'})
    engineered_features.append('Value_Tier')
    
    # ======== Prepare Sales Forecasting Features ========
    
    # Create time-based features from the transaction data
    # Group sales by date for time series analysis
    sales_ts = processed_df.groupby(pd.to_datetime(processed_df['Order Date']).dt.date)['Sales'].sum().reset_index()
    sales_ts.columns = ['Date', 'Sales']
    sales_ts['Date'] = pd.to_datetime(sales_ts['Date'])  # Ensure it's datetime
    sales_ts.set_index('Date', inplace=True)
    
    # Resample to daily frequency and fill missing values
    sales_ts = sales_ts.resample('D').sum()
    sales_ts.fillna(0, inplace=True)
    
    # Add to the processed dataframe
    sales_ts_df = sales_ts.reset_index()
    sales_ts_df.columns = ['Order_Date', 'Daily_Sales']
    
    # Add time features
    sales_ts_df['Year'] = sales_ts_df['Order_Date'].dt.year
    sales_ts_df['Month'] = sales_ts_df['Order_Date'].dt.month
    sales_ts_df['Day'] = sales_ts_df['Order_Date'].dt.day
    sales_ts_df['DayOfWeek'] = sales_ts_df['Order_Date'].dt.dayofweek
    sales_ts_df['Quarter'] = sales_ts_df['Order_Date'].dt.quarter
    sales_ts_df['WeekOfYear'] = sales_ts_df['Order_Date'].dt.isocalendar().week.astype(int)
    
    # Add lagged features
    for lag in [1, 7, 14, 28]:
        sales_ts_df[f'Sales_Lag_{lag}'] = sales_ts_df['Daily_Sales'].shift(lag)
    
    # Add rolling statistics
    for window in [7, 14, 30]:
        sales_ts_df[f'Sales_Rolling_Mean_{window}'] = sales_ts_df['Daily_Sales'].rolling(window=window).mean()
        sales_ts_df[f'Sales_Rolling_Std_{window}'] = sales_ts_df['Daily_Sales'].rolling(window=window).std()
    
    # Fill NaN values in newly created columns
    sales_ts_df.fillna(0, inplace=True)
    
    # Store both customer-level and time-series dataframes as attributes
    result_df = processed_df.copy()
    
    # Merge customer-level features back to the original dataframe
    # First extract key columns from the customer_df for the result dataframe
    key_customer_features = [
        'Customer ID', 'CLV', 'Discounted_CLV', 'Value_Tier', 'RFM_Segment', 
        'Churn_Label', 'Churn_Prediction_90d'
    ]
    
    # Get available columns
    available_columns = [col for col in key_customer_features if col in customer_df.columns]
    
    # Merge key customer metrics to the result dataframe
    if available_columns:
        result_df = pd.merge(result_df, customer_df[available_columns], on='Customer ID', how='left')
    
    # Add the time series data for forecasting
    result_df = result_df.assign(
        Sales_Forecasting_Data_Available=True,
        Sales_TS_Start_Date=sales_ts_df['Order_Date'].min(),
        Sales_TS_End_Date=sales_ts_df['Order_Date'].max(),
        Sales_TS_Num_Observations=len(sales_ts_df)
    )
    
    # Also attach the customer dataframe and sales time series for easy access
    result_df.attrs['customer_df'] = customer_df
    result_df.attrs['sales_ts_df'] = sales_ts_df
    
    return result_df, engineered_features