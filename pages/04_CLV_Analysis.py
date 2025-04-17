import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import io
from datetime import datetime
from src.alerts.risk_alerts import check_churn_risk
# Import models and utilities
from src.auth.session import requires_auth, get_user_info, get_company_id,nav_to
from src.firebase.firestore import save_dataset
from src.models.clv_model import enhanced_main_clv_analysis, EnhancedGammaGammaClvModel
class CLVAnalysisPage:
    def __init__(self):
        # Page configuration
        st.set_page_config(
            page_title="BizNexus AI | CLV Analysis", 
            page_icon="💎", 
            layout="wide"
        )
        
        # Apply custom styling
        self._apply_custom_styling()
    
    def _apply_custom_styling(self):
        """Apply professional, light pastel styling"""
        st.markdown("""
        <style>
        .main {
            background-color: #f4f6f9;
            color: #2c3e50;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e9ecef;
            border-radius: 10px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #495057;
            font-weight: 600;
            padding: 10px 15px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #dee2e6;
        }
        .stTabs [data-baseweb="tab"][data-selected="true"] {
            background-color: #4a90e2;
            color: white;
        }
        .clv-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .clv-card:hover {
            transform: scale(1.02);
        }
        .insight-container {
            border-left: 4px solid #6a89cc;
            padding-left: 15px;
            margin: 15px 0;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            text-align: center;
            height: 100%;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            color: #4a90e2;
        }
        .metric-label {
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _validate_data_availability(self):
        """
        Validate that required data is available in session state
        
        Returns:
            bool: True if data is available, False otherwise
        """
       
        
        required_keys = ['customer_df']  # Only require customer_df as essential
        
        for key in required_keys:
            if key not in st.session_state:
                st.error(f"Missing {key}. Please complete data upload step first.")
                
                # Add a button to go back to the upload page
                if st.button("Go to Data Upload", type="primary"):
                    try:
                        st.switch_page("pages/03_Upload.py")
                    except Exception as e:
                        st.error(f"Navigation error: {str(e)}")
                        try:
                            from src.auth.session import nav_to
                            nav_to("03_Upload")
                        except Exception:
                            st.error("Navigation failed. Please manually go to the Upload page.")
                
                return False
        
        # For result_df, convert it to a warning instead of an error
        if 'result_df' not in st.session_state:
            st.warning("Note: result_df is not available. Some features may be limited.")
        
        return True
    
    def _clv_configuration_options(self):
        """
        Create CLV analysis configuration options
        
        Returns:
            dict: Configuration options for CLV analysis
        """
        st.markdown("## 🔧 CLV Analysis Configuration")
        
        with st.container():
            st.markdown('<div class="clv-card">', unsafe_allow_html=True)
            
            st.markdown("### 1. CLV Calculation Basis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                time_horizon = st.selectbox(
                    "Time Horizon", 
                    [6, 12, 24, 36],
                    index=1,
                    help="Number of months to predict future customer value"
                )
            
            with col2:
                discount_rate = st.slider(
                    "Discount Rate", 
                    min_value=0.05, 
                    max_value=0.2, 
                    value=0.1, 
                    step=0.01,
                    help="Rate used to discount future cash flows to present value"
                )
            
            with col3:
                monetary_adjustments = st.checkbox(
                    "Consider Refunds & Discounts", 
                    value=False,
                    help="Include refunds and discounts when calculating CLV"
                )
            
            st.markdown("### 2. Data Preprocessing Settings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                customer_segment = st.selectbox(
                    "Customer Selection", 
                    options=["All Customers", "New Customers", "Repeat Customers"],
                    index=0,
                    help="Which customers to include in the analysis"
                )
                # Convert to expected format in model
                customer_segment = customer_segment.lower().split(" ")[0]
            
            with col2:
                outlier_handling = st.selectbox(
                    "Outlier Handling", 
                    options=["Cap at 99th Percentile", "Remove Outliers", "Keep All Values"],
                    index=0,
                    help="How to handle extreme values in CLV calculations"
                )
                # Convert to expected format in model
                outlier_handling_map = {
                    "Cap at 99th Percentile": "cap",
                    "Remove Outliers": "remove",
                    "Keep All Values": "none"
                }
                outlier_handling = outlier_handling_map[outlier_handling]
            
            with col3:
                num_segments = st.selectbox(
                    "Segmentation Depth", 
                    options=[3, 4, 5, 6],
                    index=1,
                    help="Number of customer segments to create based on CLV"
                )
            
            st.markdown("### 3. Business Context & Objectives")
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_purpose = st.multiselect(
                    "Analysis Purpose", 
                    options=["Churn Prediction", "Sales Forecasting", "Marketing Optimization", "VIP Identification"],
                    default=["Churn Prediction", "Sales Forecasting"],
                    help="Primary business objectives for this CLV analysis"
                )
            
            with col2:
                top_percentile = st.slider(
                    "High-Value Customer Threshold (%)", 
                    min_value=5, 
                    max_value=30, 
                    value=10, 
                    step=5,
                    help="Define top percentile for high-value customers"
                )
            
            # Advanced Filtering Section
            st.markdown("### 4. Advanced Filtering")
            col1, col2, col3 = st.columns(3)
            
            selected_regions = []
            selected_channels = []
            selected_customer_types = []
            
            # Use session state to access customer_df
            if 'customer_df' in st.session_state:
                customer_df = st.session_state.customer_df
                
                with col1:
                    if 'Region' in customer_df.columns:
                        regions = customer_df['Region'].unique().tolist()
                        selected_regions = st.multiselect(
                            "Select Regions", 
                            options=regions, 
                            default=regions
                        )
                
                with col2:
                    if 'Acquisition_Channel' in customer_df.columns:
                        channels = customer_df['Acquisition_Channel'].unique().tolist()
                        selected_channels = st.multiselect(
                            "Select Acquisition Channels", 
                            options=channels, 
                            default=channels
                        )
                
                with col3:
                    if 'Customer_Type' in customer_df.columns:
                        customer_types = customer_df['Customer_Type'].unique().tolist()
                        selected_customer_types = st.multiselect(
                            "Select Customer Types", 
                            options=customer_types, 
                            default=customer_types
                        )
            
            # Business-Specific Model Optimization
            st.markdown("### 5. Business-Specific Model Optimization")
            col1, col2 = st.columns(2)

            with col1:
                business_type = st.selectbox(
                    "Business Type", 
                    options=["Retail", "E-commerce", "Subscription", "B2B", "SaaS"],
                    index=0,
                    help="Optimize CLV parameters for your specific business model"
                )

            with col2:
                customer_pattern = st.selectbox(
                    "Customer Behavior Pattern", 
                    options=["Standard", "Seasonal", "High Churn", "Loyal"],
                    index=0,
                    help="Fine-tune model for specific customer behavior patterns"
                )

            # Display customization effects (add this after customer_pattern selection)
            if business_type != "Retail" or customer_pattern != "Standard":
                st.markdown("#### 📊 Parameter Customization Effects")
                
                # Time horizon adjustment
                th_adj = 0
                if business_type == "Subscription" or business_type == "SaaS":
                    th_adj += 12
                elif business_type == "B2B":
                    th_adj += 24
                
                if customer_pattern == "Loyal":
                    th_adj += 12
                elif customer_pattern == "High Churn":
                    th_adj -= 6
                
                # Discount rate adjustment
                dr_adj = 0
                if business_type == "Retail" or business_type == "E-commerce":
                    dr_adj += 0.02
                elif business_type == "Subscription" or business_type == "SaaS":
                    dr_adj -= 0.02
                
                if customer_pattern == "High Churn":
                    dr_adj += 0.05
                elif customer_pattern == "Loyal":
                    dr_adj -= 0.02
                
                # Display adjustments
                effects_col1, effects_col2 = st.columns(2)
                
                with effects_col1:
                    st.info(f"Time Horizon: +{th_adj} months" if th_adj > 0 else f"Time Horizon: {th_adj} months")
                
                with effects_col2:
                    st.info(f"Discount Rate: +{dr_adj:.2f}" if dr_adj > 0 else f"Discount Rate: {dr_adj:.2f}")
                
                # Update config based on business type
                if th_adj != 0:
                    time_horizon = min(max(time_horizon + th_adj, 6), 60)
                
                if dr_adj != 0:
                    discount_rate = min(max(discount_rate + dr_adj, 0.05), 0.2)
                    
            st.markdown('</div>', unsafe_allow_html=True)
            
            
            
            return {
                'time_horizon': time_horizon,
                'discount_rate': discount_rate,
                'monetary_adjustments': monetary_adjustments,
                'customer_segment': customer_segment,
                'outlier_handling': outlier_handling,
                'num_segments': num_segments,
                'analysis_purpose': analysis_purpose,
                'top_percentile': top_percentile,
                'selected_regions': selected_regions,
                'selected_channels': selected_channels,
                'selected_customer_types': selected_customer_types,
                'business_type': business_type,
                'customer_pattern': customer_pattern
            }

    def _key_insights_and_visualization(self, clv_results, config):
        """
        Show key insights and visualizations based on analysis purpose
        
        Args:
            clv_results (pd.DataFrame): CLV results dataframe
            config (dict): Configuration options
        
        Returns:
            dict: Configuration options passed through
        """
        st.markdown("## 🔍 Key Business Insights")
        
        analysis_purposes = config['analysis_purpose']
        
        # Find the primary CLV column to use for analysis
        clv_column_priority = ['Predicted_CLV', 'CLV', 'CLV_Adjusted', 'Discounted_CLV']
        clv_column = next((col for col in clv_column_priority if col in clv_results.columns), None)
        
        if not clv_column:
            st.error("No CLV column found in results. Cannot perform business insights analysis.")
            return config
            
        # Find the churn probability column
        churn_column_priority = ['Churn_Probability', 'Churn_Prediction_90d', 'Churn_Label']
        churn_column = next((col for col in churn_column_priority if col in clv_results.columns), None)
        
        # Find the segment column
        segment_column = 'Value_Tier' if 'Value_Tier' in clv_results.columns else 'CLV_Segment'
        
        with st.container():
            st.markdown('<div class="clv-card">', unsafe_allow_html=True)
            
            if "Churn Prediction" in analysis_purposes and churn_column:
                st.markdown("### 🚨 Churn Risk Analysis")
                
                # Find high-value customers at risk of churning
                high_value_tier = 'Premium' if segment_column in clv_results.columns else None
                
                if high_value_tier:
                    high_value_at_risk = clv_results[
                        (clv_results[segment_column] == high_value_tier) & 
                        (clv_results[churn_column] > 0.5)
                    ]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### High-Value Customers at Risk")
                        
                        if len(high_value_at_risk) > 0:
                            # Determine columns to display for at-risk customers
                            at_risk_cols = []
                            for col in ['Customer ID', 'Customer Name', clv_column, churn_column]:
                                if col in clv_results.columns:
                                    at_risk_cols.append(col)
                            
                            # Create formatting dictionary
                            format_dict = {}
                            if clv_column in at_risk_cols:
                                format_dict[clv_column] = '${:.2f}'
                            if churn_column in at_risk_cols:
                                format_dict[churn_column] = '{:.1%}'
                            
                            st.dataframe(
                                high_value_at_risk[at_risk_cols].head(10).style.format(format_dict),
                                use_container_width=True
                            )
                            
                            potential_loss = high_value_at_risk[clv_column].sum()
                            st.markdown(f"<p class='insight-container'>Potential revenue at risk: <strong>${potential_loss:,.2f}</strong></p>", unsafe_allow_html=True)
                        else:
                            st.info("No high-value customers currently at high risk of churning.")
                    
                    with col2:
                        # Scatterplot of CLV vs Churn Probability
                        fig_scatter = px.scatter(
                            clv_results,
                            x=churn_column,
                            y=clv_column,
                            color=segment_column if segment_column in clv_results.columns else None,
                            title='CLV vs Churn Risk',
                            labels={
                                churn_column: 'Churn Probability', 
                                clv_column: 'Customer Lifetime Value ($)'
                            }
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
            if "Sales Forecasting" in analysis_purposes:
                st.markdown("### 📈 Sales Forecast Based on CLV")
                
                # Calculate monthly forecast from CLV
                if 'Monthly_Forecast' in clv_results.columns:
                    monthly_forecast_column = 'Monthly_Forecast'
                else:
                    clv_results['Monthly_Forecast'] = clv_results[clv_column] / config['time_horizon']
                    monthly_forecast_column = 'Monthly_Forecast'
                
                # Total monthly forecast
                total_monthly_forecast = clv_results[monthly_forecast_column].sum()
                total_forecast = clv_results[clv_column].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Monthly forecast by segment
                    if segment_column in clv_results.columns:
                        segment_forecast = clv_results.groupby(segment_column)[monthly_forecast_column].sum().reset_index()
                        
                        fig_forecast = px.bar(
                            segment_forecast,
                            x=segment_column,
                            y=monthly_forecast_column,
                            title='Monthly Revenue Forecast by Segment',
                            labels={
                                monthly_forecast_column: 'Monthly Revenue ($)', 
                                segment_column: 'Customer Segment'
                            },
                            color=segment_column,
                            text=monthly_forecast_column
                        )
                        fig_forecast.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig_forecast, use_container_width=True)
                
                with col2:
                    # Forecast metrics
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">${total_monthly_forecast:,.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Monthly Revenue Forecast</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="metric-card" style="margin-top:20px;">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">${total_forecast:,.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Total Revenue Forecast ({config["time_horizon"]} months)</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return config
    
    def _perform_clv_analysis(self, customer_df, config):
        """
        Perform comprehensive CLV analysis and provide local download option
        
        Args:
            customer_df (pd.DataFrame): Customer dataframe
            config (dict): Configuration options for CLV analysis
        
        Returns:
            dict: CLV analysis results with standardized structure
        """
        st.markdown("## 📊 Running CLV Analysis")
        
        # Default return structure to ensure consistent return
        default_return = {
            'success': False,
            'error': 'Unknown error occurred',
            'clv_results': None,
            'descriptive_stats': None,
            'model': None
        }
        
        try:
            with st.spinner("Calculating Customer Lifetime Value..."):
                # Apply business-specific optimizations if selected
                if config.get('business_type', 'Retail') != 'Retail' or config.get('customer_pattern', 'Standard') != 'Standard':
                    # Add a checkbox to apply the optimizations
                    apply_optimizations = st.checkbox("Apply Business-Specific Optimizations", value=True)
                    
                    if apply_optimizations:
                        # Calculate optimized parameters based on business type and customer pattern
                        business_type = config.get('business_type', 'Retail')
                        customer_pattern = config.get('customer_pattern', 'Standard')
                        
                        # Time horizon adjustments
                        th_adj = 0
                        if business_type == "Subscription" or business_type == "SaaS":
                            th_adj += 12
                        elif business_type == "B2B":
                            th_adj += 24
                        
                        if customer_pattern == "Loyal":
                            th_adj += 12
                        elif customer_pattern == "High Churn":
                            th_adj -= 6
                        
                        # Discount rate adjustments
                        dr_adj = 0
                        if business_type == "Retail" or business_type == "E-commerce":
                            dr_adj += 0.02
                        elif business_type == "Subscription" or business_type == "SaaS":
                            dr_adj -= 0.02
                        
                        if customer_pattern == "High Churn":
                            dr_adj += 0.05
                        elif customer_pattern == "Loyal":
                            dr_adj -= 0.02
                        
                        # Update config
                        config['time_horizon'] = min(max(config['time_horizon'] + th_adj, 6), 60)
                        config['discount_rate'] = min(max(config['discount_rate'] + dr_adj, 0.05), 0.2)
                        
                        # Update outlier handling based on business type
                        if business_type == "B2B":
                            config['outlier_handling'] = "none"  # B2B often has natural high-value outliers
                        
                        # Show the optimized parameters
                        st.info(f"Optimized Parameters: Time Horizon = {config['time_horizon']} months, Discount Rate = {config['discount_rate']:.2f}")
                
                # Perform CLV analysis with the configured parameters
                clv_analysis = enhanced_main_clv_analysis(
                    customer_df, 
                    time_horizon=config['time_horizon'], 
                    discount_rate=config['discount_rate'],
                    outlier_handling=config['outlier_handling'],
                    monetary_adjustments=config['monetary_adjustments'],
                    customer_segment=config['customer_segment'],
                    num_segments=config['num_segments'],
                    perform_evaluation=True,
                    train_test_split_ratio=0.2
                )
                
                # Check if analysis was successful
                if not clv_analysis.get('success', False):
                    # If the analysis itself failed
                    default_return['error'] = clv_analysis.get('error', 'CLV analysis failed')
                    st.error(default_return['error'])
                    return default_return
                
                # Prepare the return dictionary
                return_dict = {
                    'success': True,
                    'clv_results': clv_analysis.get('clv_results'),
                    'descriptive_stats': clv_analysis.get('descriptive_stats', {}),
                    'model': clv_analysis.get('model'),
                    'clv_report': clv_analysis.get('clv_report', {})
                }
                
                # CLV Report Section
                if return_dict['clv_report']:
                    st.markdown("## 📑 CLV Analysis Report")
                    
                    report = return_dict['clv_report']
                    
                    # Summary section
                    if 'summary' in report:
                        summary = report['summary']
                        
                        st.markdown("### Summary Statistics")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            st.metric(
                                "Total Customers", 
                                f"{summary.get('total_customers', 0):,}"
                            )
                        
                        with summary_col2:
                            st.metric(
                                "Total CLV", 
                                f"${summary.get('total_clv', 0):,.2f}"
                            )
                        
                        with summary_col3:
                            st.metric(
                                "Average CLV", 
                                f"${summary.get('average_clv', 0):,.2f}"
                            )
                    
                    # Segment contribution
                    if 'segment_contribution' in report:
                        st.markdown("### Customer Segment Value")
                        
                        segment_data = pd.DataFrame({
                            'Segment': list(report['segment_contribution'].keys()),
                            'Contribution %': list(report['segment_contribution'].values())
                        })
                        
                        fig_segment = px.bar(
                            segment_data,
                            x='Segment',
                            y='Contribution %',
                            text='Contribution %',
                            color='Segment',
                            title='Revenue Contribution by Customer Segment'
                        )
                        fig_segment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig_segment, use_container_width=True)
                    
                    # Actionable recommendations
                    if 'recommendations' in report and report['recommendations']:
                        st.markdown("### 🚀 Actionable Recommendations")
                        
                        for i, recommendation in enumerate(report['recommendations']):
                            st.markdown(f"{i+1}. {recommendation}")
                
                return return_dict
        
        except Exception as e:
            # Catch any unexpected errors
            st.error(f"CLV Analysis failed: {str(e)}")
            st.exception(e)
            
            # Update the default return with the specific error
            default_return['error'] = str(e)
            return default_return
    def _show_dataset_preview(self, clv_results):
        """
        Show a preview of the CLV results dataset
        
        Args:
            clv_results (pd.DataFrame): CLV results dataframe
        """
        st.markdown("## 📋 CLV Analysis Results")
        
        with st.container():
            st.markdown('<div class="clv-card">', unsafe_allow_html=True)
            
            # Map display column names to actual column names in the dataframe
            column_mapping = {
                'Customer ID': 'Customer ID',
                'Customer Name': 'Customer Name',
                'Predicted CLV': 'Predicted_CLV',
                'CLV': 'CLV',
                'Adjusted CLV': 'CLV_Adjusted',
                'Discounted CLV': 'Discounted_CLV',
                'Value Tier': 'Value_Tier',
                'Churn Probability': 'Churn_Probability',
                'Churn Prediction': 'Churn_Prediction_90d'
            }
            
            # Determine columns to display based on what's available
            display_cols = []
            for display_name, col_name in column_mapping.items():
                if col_name in clv_results.columns:
                    display_cols.append(col_name)
            
            # Ensure Customer ID is always first if available
            if 'Customer ID' in display_cols:
                display_cols.remove('Customer ID')
                display_cols.insert(0, 'Customer ID')
                
            # Ensure Customer Name is second if available
            if 'Customer Name' in display_cols:
                display_cols.remove('Customer Name')
                display_cols.insert(1 if 'Customer ID' in display_cols else 0, 'Customer Name')
                
            # Create formatting dict for numeric columns
            format_dict = {}
            for col in display_cols:
                if col in ['CLV', 'Predicted_CLV', 'CLV_Adjusted', 'Discounted_CLV'] and col in clv_results.columns:
                    format_dict[col] = '${:.2f}'
                elif col in ['Churn_Probability', 'Churn_Prediction_90d'] and col in clv_results.columns:
                    format_dict[col] = '{:.1%}'
            
            # Show the data with formatting
            st.dataframe(
                clv_results[display_cols].style.format(format_dict),
                use_container_width=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    
   
    def _descriptive_analysis(self, clv_results, descriptive_stats):
        """
        Perform comprehensive descriptive analysis of CLV results with advanced visualizations
        
        Args:
            clv_results (pd.DataFrame): CLV results dataframe
            descriptive_stats (dict): Descriptive statistics
            clv_model (EnhancedGammaGammaClvModel, optional): The CLV model object for visualizations
        """
        st.markdown("## 📊 Comprehensive CLV Analysis")
        
        # Find the primary CLV column to use for analysis
        clv_column_priority = ['Predicted_CLV', 'CLV', 'CLV_Adjusted', 'Discounted_CLV']
        clv_column = next((col for col in clv_column_priority if col in clv_results.columns), None)
        
        if not clv_column:
            st.error("No CLV column found in results. Cannot perform descriptive analysis.")
            return
        
        # Create container for filters and recommendations
        filter_col, recommendations_col = st.columns([3, 2])
        
        with filter_col:
            # Interactive Filters
            st.markdown("### 🔍 CLV Analysis Filters")
            
            # Create filter columns
            filter_row1 = st.columns(3)
            filter_row2 = st.columns(3)
            
            # Acquisition Channel Filter
            with filter_row1[0]:
                if 'Acquisition_Channel' in clv_results.columns:
                    channels = clv_results['Acquisition_Channel'].unique().tolist()
                    selected_channels = st.multiselect(
                        "Acquisition Channel", 
                        options=channels, 
                        default=channels
                    )
                    clv_results_filtered = clv_results[clv_results['Acquisition_Channel'].isin(selected_channels)]
                else:
                    clv_results_filtered = clv_results
            
            # Region Filter
            with filter_row1[1]:
                if 'Region' in clv_results.columns:
                    regions = clv_results['Region'].unique().tolist()
                    selected_regions = st.multiselect(
                        "Region", 
                        options=regions, 
                        default=regions
                    )
                    clv_results_filtered = clv_results_filtered[clv_results_filtered['Region'].isin(selected_regions)]
            
            # Customer Type Filter
            with filter_row1[2]:
                if 'Customer_Type' in clv_results.columns:
                    customer_types = clv_results['Customer_Type'].unique().tolist()
                    selected_types = st.multiselect(
                        "Customer Type", 
                        options=customer_types, 
                        default=customer_types
                    )
                    clv_results_filtered = clv_results_filtered[clv_results_filtered['Customer_Type'].isin(selected_types)]
            
            # Tenure Filter
            with filter_row2[0]:
                if 'Tenure' in clv_results.columns:
                    min_tenure = clv_results['Tenure'].min()
                    max_tenure = clv_results['Tenure'].max()
                    tenure_range = st.slider(
                        "Customer Tenure (Years)", 
                        min_value=min_tenure, 
                        max_value=max_tenure, 
                        value=(min_tenure, max_tenure)
                    )
                    clv_results_filtered = clv_results_filtered[
                        (clv_results_filtered['Tenure'] >= tenure_range[0]) & 
                        (clv_results_filtered['Tenure'] <= tenure_range[1])
                    ]
        
        # Generate Context-Aware Recommendations
        with recommendations_col:
            st.markdown("### 🚀 Contextual Recommendations")
            recommendations = self._generate_context_recommendations(clv_results_filtered, clv_column)
            
            # Overall Recommendations
            st.markdown("#### Overall Insights")
            for insight in recommendations['overall']:
                st.markdown(f"- {insight}")
            
            # Targeting Recommendations
            st.markdown("#### Targeting Strategies")
            for recommendation in recommendations['targeting']:
                st.markdown(f"- {recommendation}")
            
            # Retention Recommendations
            st.markdown("#### Retention Focus")
            for recommendation in recommendations['retention']:
                st.markdown(f"- {recommendation}")
            
            # Acquisition Recommendations
            st.markdown("#### Acquisition Insights")
            for recommendation in recommendations['acquisition']:
                st.markdown(f"- {recommendation}")
        
        # Visualization Tabs
        visualization_tabs = st.tabs([
            "Distribution Analysis", 
            "CLV Trends", 
            "Customer Metrics", 
            "Revenue Impact", 
            "RFM Analysis"
        ])
        
        # Distribution Analysis Tab
        with visualization_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of CLV with Pareto Principle Overlay
                fig_hist = px.histogram(
                    clv_results_filtered, 
                    x=clv_column,
                    nbins=30,
                    title='CLV Distribution with Pareto Principle',
                    labels={clv_column: 'Customer Lifetime Value ($)'},
                    color_discrete_sequence=['#4a90e2']
                )
                # Add Pareto Principle line
                pareto_threshold = np.percentile(clv_results_filtered[clv_column], 80)
                fig_hist.add_vline(
                    x=pareto_threshold, 
                    line_dash="dash", 
                    line_color="red", 
                    annotation_text="80/20 Rule", 
                    annotation_position="top right"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot with outlier analysis
                fig_box = px.box(
                    clv_results_filtered,
                    y=clv_column,
                    title='CLV Box Plot (Outlier Analysis)',
                    labels={clv_column: 'Customer Lifetime Value ($)'},
                    color_discrete_sequence=['#4a90e2']
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # CLV Trends Tab
        with visualization_tabs[1]:
            # Cohort Analysis or Time-based Trends
            if 'Signup_Date' in clv_results_filtered.columns:
                # Convert signup date to month
                clv_results_filtered['Signup_Month'] = pd.to_datetime(clv_results_filtered['Signup_Date']).dt.strftime('%Y-%m')
                
                # Monthly CLV Trend
                monthly_clv = clv_results_filtered.groupby('Signup_Month')[clv_column].agg(['mean', 'count']).reset_index()
                
                fig_monthly_trend = px.line(
                    monthly_clv, 
                    x='Signup_Month', 
                    y='mean',
                    title='Average CLV by Signup Month',
                    labels={'mean': 'Average CLV ($)', 'Signup_Month': 'Signup Month'},
                    hover_data={'count': ':.0f'}
                )
                fig_monthly_trend.update_traces(mode='lines+markers')
                st.plotly_chart(fig_monthly_trend, use_container_width=True)
            else:
                st.info("No time-based data available for trend analysis.")
        
        # Customer Metrics Tab
        with visualization_tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                # CLV vs Churn Probability
                if 'Churn_Probability' in clv_results_filtered.columns:
                    fig_churn = px.scatter(
                        clv_results_filtered,
                        x=clv_column,
                        y='Churn_Probability',
                        color='Customer_Type' if 'Customer_Type' in clv_results_filtered.columns else None,
                        title='CLV vs Churn Probability',
                        labels={
                            clv_column: 'Customer Lifetime Value ($)', 
                            'Churn_Probability': 'Churn Probability'
                        }
                    )
                    st.plotly_chart(fig_churn, use_container_width=True)
                else:
                    st.info("Churn probability data not available.")
            
            with col2:
                # CLV vs Customer Acquisition Cost
                if 'Customer_Acquisition_Cost' in clv_results_filtered.columns:
                    fig_cac = px.scatter(
                        clv_results_filtered,
                        x=clv_column,
                        y='Customer_Acquisition_Cost',
                        color='Acquisition_Channel' if 'Acquisition_Channel' in clv_results_filtered.columns else None,
                        title='CLV vs Customer Acquisition Cost',
                        labels={
                            clv_column: 'Customer Lifetime Value ($)', 
                            'Customer_Acquisition_Cost': 'Acquisition Cost ($)'
                        }
                    )
                    st.plotly_chart(fig_cac, use_container_width=True)
                else:
                    st.info("Customer acquisition cost data not available.")
        
        # Revenue Impact Tab
        with visualization_tabs[3]:
            # Segment Revenue Contribution
            segment_column = 'Value_Tier' if 'Value_Tier' in clv_results_filtered.columns else 'CLV_Segment'
            
            if segment_column in clv_results_filtered.columns:
                # Revenue Contribution by Segment
                segment_revenue = clv_results_filtered.groupby(segment_column)[clv_column].sum().reset_index()
                
                fig_revenue_contrib = px.pie(
                    segment_revenue,
                    values=clv_column,
                    names=segment_column,
                    title='Revenue Contribution by Customer Segment',
                    hole=0.3
                )
                st.plotly_chart(fig_revenue_contrib, use_container_width=True)
                
                # Retention Rate by Segment
                if 'Retention_Rate' in clv_results_filtered.columns:
                    retention_by_segment = clv_results_filtered.groupby(segment_column)['Retention_Rate'].mean().reset_index()
                    
                    fig_retention = px.bar(
                        retention_by_segment,
                        x=segment_column,
                        y='Retention_Rate',
                        title='Retention Rate by Customer Segment',
                        labels={'Retention_Rate': 'Average Retention Rate'}
                    )
                    st.plotly_chart(fig_retention, use_container_width=True)
                else:
                    st.info("Retention rate data not available.")
            else:
                st.info("No customer segmentation data available.")
        
        # RFM Analysis Tab
        with visualization_tabs[4]:
            # Perform RFM Analysis
            rfm_results = self._generate_rfm_analysis(clv_results_filtered)
            
            if rfm_results is not None:
                # RFM Segment Distribution
                segment_distribution = rfm_results['RFM_Segment'].value_counts()
                
                fig_rfm_segments = px.pie(
                    values=segment_distribution.values,
                    names=segment_distribution.index,
                    title='Customer Segments by RFM Analysis',
                    hole=0.3
                )
                st.plotly_chart(fig_rfm_segments, use_container_width=True)
                
                # RFM Scatter Plot
                fig_rfm_scatter = px.scatter(
                    rfm_results,
                    x='Frequency_Score',
                    y='Monetary_Score',
                    color='RFM_Segment',
                    size='RFM_Score',
                    hover_data=[clv_column],
                    title='RFM Segmentation',
                    labels={
                        'Frequency_Score': 'Frequency Score', 
                        'Monetary_Score': 'Monetary Score'
                    }
                )
                st.plotly_chart(fig_rfm_scatter, use_container_width=True)
            else:
                st.warning("Insufficient data for RFM analysis. Please ensure you have columns for purchase date, number of purchases, and total spend.")
        
    
    def _generate_context_recommendations(self, clv_results_filtered, clv_column):
        """
        Generate dynamic recommendations based on filtered CLV data
        
        Args:
            clv_results_filtered (pd.DataFrame): Filtered CLV results
            clv_column (str): Name of the CLV column
        
        Returns:
            dict: Contextual recommendations
        """
        recommendations = {
            'overall': [],
            'targeting': [],
            'retention': [],
            'acquisition': []
        }
        
        # Overall Performance Insights
        total_customers = len(clv_results_filtered)
        
        if total_customers == 0:
            recommendations['overall'].append("No customers match the current filters.")
            return recommendations
        
        total_clv = clv_results_filtered[clv_column].sum()
        mean_clv = clv_results_filtered[clv_column].mean()
        
        recommendations['overall'].append(f"Total Customers: {total_customers}")
        recommendations['overall'].append(f"Total Customer Lifetime Value: ${total_clv:,.2f}")
        recommendations['overall'].append(f"Average Customer Lifetime Value: ${mean_clv:,.2f}")
        
        # Segmentation Recommendations
        if 'Value_Tier' in clv_results_filtered.columns:
            segment_breakdown = clv_results_filtered['Value_Tier'].value_counts(normalize=True)
            
            if 'High' in segment_breakdown:
                high_value_pct = segment_breakdown['High'] * 100
                recommendations['targeting'].append(f"{high_value_pct:.1f}% of customers are in the High-Value segment")
                
                if high_value_pct < 20:
                    recommendations['targeting'].append("Consider implementing a VIP program to elevate more customers")
                elif high_value_pct > 50:
                    recommendations['targeting'].append("Your customer base is highly valuable. Focus on retention strategies")
        
        # Churn Risk Analysis
        if 'Churn_Probability' in clv_results_filtered.columns:
            high_churn_risk = clv_results_filtered[clv_results_filtered['Churn_Probability'] > 0.5]
            high_churn_pct = len(high_churn_risk) / total_customers * 100
            
            recommendations['retention'].append(f"{high_churn_pct:.1f}% of customers have high churn risk")
            
            if high_churn_pct > 30:
                recommendations['retention'].extend([
                    "Urgent: Develop targeted retention campaigns",
                    "Analyze common characteristics of high-risk customers",
                    "Implement proactive engagement strategies"
                ])
        
        # Acquisition Channel Insights
        if 'Acquisition_Channel' in clv_results_filtered.columns:
            channel_performance = clv_results_filtered.groupby('Acquisition_Channel')[clv_column].agg(['mean', 'count'])
            best_channel = channel_performance['mean'].idxmax()
            
            recommendations['acquisition'].extend([
                f"Best performing acquisition channel: {best_channel}",
                "Channel Performance (Avg CLV): " + 
                ", ".join([f"{channel}: ${avg:,.2f}" for channel, avg in channel_performance['mean'].items()])
            ])
        
        # Regional Performance
        if 'Region' in clv_results_filtered.columns:
            region_performance = clv_results_filtered.groupby('Region')[clv_column].agg(['mean', 'count'])
            top_region = region_performance['mean'].idxmax()
            
            recommendations['acquisition'].extend([
                f"Top performing region: {top_region}",
                "Regional Performance (Avg CLV): " + 
                ", ".join([f"{region}: ${avg:,.2f}" for region, avg in region_performance['mean'].items()])
            ])
        
        # Tenure Impact
        if 'Tenure' in clv_results_filtered.columns:
            tenure_bins = pd.cut(clv_results_filtered['Tenure'], 
                                bins=[0, 1, 2, 3, float('inf')], 
                                labels=['0-1 year', '1-2 years', '2-3 years', '3+ years'])
            tenure_clv = clv_results_filtered.groupby(tenure_bins)[clv_column].mean()
            
            recommendations['retention'].append("CLV by Customer Tenure:")
            for tenure, avg_clv in tenure_clv.items():
                recommendations['retention'].append(f"{tenure}: ${avg_clv:,.2f}")
        
        return recommendations

    def _generate_rfm_analysis(self, clv_results_filtered):
        """
        Generate RFM (Recency, Frequency, Monetary) analysis
        
        Args:
            clv_results_filtered (pd.DataFrame): Filtered CLV results
        
        Returns:
            pd.DataFrame or None: RFM scored dataframe or None if insufficient data
        """
        # Check for required columns
        rfm_columns = ['Date', 'NumPurchases', 'TotalSpend']
        alternative_columns = [
            ['First_Purchase_Date', 'Num_of_Purchases', 'Total_Spend'],
            ['Last_Purchase_Date', 'Num_of_Purchases', 'Total_Spend']
        ]
        
        # Try to find matching column sets
        found_columns = None
        for column_set in alternative_columns:
            if all(col in clv_results_filtered.columns for col in column_set):
                found_columns = column_set
                break
        
        # If no matching columns found, return None
        if found_columns is None:
            st.warning("RFM Analysis requires columns representing: Purchase Date, Number of Purchases, and Total Spend")
            return None
        
        # Rename columns to standard names
        clv_results_filtered = clv_results_filtered.rename(columns={
            found_columns[0]: 'Date',
            found_columns[1]: 'NumPurchases',
            found_columns[2]: 'TotalSpend'
        })
        
        # Recency Calculation (days since last purchase)
        # Use Last_Purchase_Date for recency calculation
        current_date = pd.Timestamp.now()
        clv_results_filtered['Recency'] = (current_date - pd.to_datetime(clv_results_filtered['Last_Purchase_Date'])).dt.days
        
        # RFM Scoring Functions
        def score_recency(x):
            if x <= 30: return 4
            elif x <= 90: return 3
            elif x <= 180: return 2
            else: return 1
        
        def score_frequency(x):
            if x >= 10: return 4
            elif x >= 5: return 3
            elif x >= 2: return 2
            else: return 1
        
        def score_monetary(x):
            if x >= 1000: return 4
            elif x >= 500: return 3
            elif x >= 200: return 2
            else: return 1
        
        # Apply scoring
        clv_results_filtered['Recency_Score'] = clv_results_filtered['Recency'].apply(score_recency)
        clv_results_filtered['Frequency_Score'] = clv_results_filtered['NumPurchases'].apply(score_frequency)
        clv_results_filtered['Monetary_Score'] = clv_results_filtered['TotalSpend'].apply(score_monetary)
        
        # Combine RFM Scores
        clv_results_filtered['RFM_Score'] = (
            clv_results_filtered['Recency_Score'] * 100 + 
            clv_results_filtered['Frequency_Score'] * 10 + 
            clv_results_filtered['Monetary_Score']
        )
        
        # Customer Segmentation based on RFM
        def categorize_customer(score):
            if score >= 444: return 'Champions'
            elif score >= 434: return 'Loyal'
            elif score >= 424: return 'Potential Loyalist'
            elif score >= 414: return 'New Customer'
            elif score >= 404: return 'Promising'
            elif score >= 394: return 'At Risk'
            else: return 'Hibernating'
        
        clv_results_filtered['RFM_Segment'] = clv_results_filtered['RFM_Score'].apply(categorize_customer)
        
        return clv_results_filtered

    def _descriptive_analysis(self, clv_results, descriptive_stats):
        """
        Perform comprehensive descriptive analysis of CLV results with advanced visualizations
        
        Args:
            clv_results (pd.DataFrame): CLV results dataframe
            descriptive_stats (dict): Descriptive statistics
        """
        import streamlit as st
        import plotly.express as px
        import numpy as np
        import pandas as pd
        
        st.markdown("## 📊 Comprehensive CLV Analysis")
        
        # Find the primary CLV column to use for analysis
        clv_column_priority = ['Predicted_CLV', 'CLV', 'CLV_Adjusted', 'Discounted_CLV']
        clv_column = next((col for col in clv_column_priority if col in clv_results.columns), None)
        
        if not clv_column:
            st.error("No CLV column found in results. Cannot perform descriptive analysis.")
            return
        
        # Create container for filters and recommendations
        filter_col, recommendations_col = st.columns([3, 2])
        
        with filter_col:
            # Interactive Filters
            st.markdown("### 🔍 CLV Analysis Filters")
            
            # Create filter columns
            filter_row1 = st.columns(3)
            filter_row2 = st.columns(3)
            
            # Acquisition Channel Filter
            with filter_row1[0]:
                if 'Acquisition_Channel' in clv_results.columns:
                    channels = clv_results['Acquisition_Channel'].unique().tolist()
                    selected_channels = st.multiselect(
                        "Acquisition Channel", 
                        options=channels, 
                        default=channels
                    )
                    clv_results_filtered = clv_results[clv_results['Acquisition_Channel'].isin(selected_channels)]
                else:
                    clv_results_filtered = clv_results
            
            # Region Filter
            with filter_row1[1]:
                if 'Region' in clv_results.columns:
                    regions = clv_results['Region'].unique().tolist()
                    selected_regions = st.multiselect(
                        "Region", 
                        options=regions, 
                        default=regions
                    )
                    clv_results_filtered = clv_results_filtered[clv_results_filtered['Region'].isin(selected_regions)]
            
            # Customer Type Filter
            with filter_row1[2]:
                if 'Customer_Type' in clv_results.columns:
                    customer_types = clv_results['Customer_Type'].unique().tolist()
                    selected_types = st.multiselect(
                        "Customer Type", 
                        options=customer_types, 
                        default=customer_types
                    )
                    clv_results_filtered = clv_results_filtered[clv_results_filtered['Customer_Type'].isin(selected_types)]
            
            # Tenure Filter
            with filter_row2[0]:
                if 'Tenure' in clv_results.columns:
                    min_tenure = clv_results['Tenure'].min()
                    max_tenure = clv_results['Tenure'].max()
                    tenure_range = st.slider(
                        "Customer Tenure (Years)", 
                        min_value=min_tenure, 
                        max_value=max_tenure, 
                        value=(min_tenure, max_tenure)
                    )
                    clv_results_filtered = clv_results_filtered[
                        (clv_results_filtered['Tenure'] >= tenure_range[0]) & 
                        (clv_results_filtered['Tenure'] <= tenure_range[1])
                    ]
        
        # Generate Context-Aware Recommendations
        with recommendations_col:
            st.markdown("### 🚀 Contextual Recommendations")
            recommendations = self._generate_context_recommendations(clv_results_filtered, clv_column)
            
            # Overall Recommendations
            st.markdown("#### Overall Insights")
            for insight in recommendations['overall']:
                st.markdown(f"- {insight}")
            
            # Targeting Recommendations
            st.markdown("#### Targeting Strategies")
            for recommendation in recommendations['targeting']:
                st.markdown(f"- {recommendation}")
            
            # Retention Recommendations
            st.markdown("#### Retention Focus")
            for recommendation in recommendations['retention']:
                st.markdown(f"- {recommendation}")
            
            # Acquisition Recommendations
            st.markdown("#### Acquisition Insights")
            for recommendation in recommendations['acquisition']:
                st.markdown(f"- {recommendation}")
        
        # Visualization Tabs
        visualization_tabs = st.tabs([
            "Distribution Analysis", 
            "CLV Trends", 
            "Customer Metrics", 
            "Revenue Impact", 
            "RFM Analysis"
        ])
        
        # Distribution Analysis Tab
        with visualization_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of CLV with Pareto Principle Overlay
                fig_hist = px.histogram(
                    clv_results_filtered, 
                    x=clv_column,
                    nbins=30,
                    title='CLV Distribution with Pareto Principle',
                    labels={clv_column: 'Customer Lifetime Value ($)'},
                    color_discrete_sequence=['#4a90e2']
                )
                # Add Pareto Principle line
                pareto_threshold = np.percentile(clv_results_filtered[clv_column], 80)
                fig_hist.add_vline(
                    x=pareto_threshold, 
                    line_dash="dash", 
                    line_color="red", 
                    annotation_text="80/20 Rule", 
                    annotation_position="top right"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot with outlier analysis
                fig_box = px.box(
                    clv_results_filtered,
                    y=clv_column,
                    title='CLV Box Plot (Outlier Analysis)',
                    labels={clv_column: 'Customer Lifetime Value ($)'},
                    color_discrete_sequence=['#4a90e2']
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # CLV Trends Tab
        with visualization_tabs[1]:
            # Cohort Analysis or Time-based Trends
            if 'Signup_Date' in clv_results_filtered.columns:
                # Convert signup date to month
                clv_results_filtered['Signup_Month'] = pd.to_datetime(clv_results_filtered['Signup_Date']).dt.strftime('%Y-%m')
                
                # Monthly CLV Trend
                monthly_clv = clv_results_filtered.groupby('Signup_Month')[clv_column].agg(['mean', 'count']).reset_index()
                
                fig_monthly_trend = px.line(
                    monthly_clv, 
                    x='Signup_Month', 
                    y='mean',
                    title='Average CLV by Signup Month',
                    labels={'mean': 'Average CLV ($)', 'Signup_Month': 'Signup Month'},
                    hover_data={'count': ':.0f'}
                )
                fig_monthly_trend.update_traces(mode='lines+markers')
                st.plotly_chart(fig_monthly_trend, use_container_width=True)
            else:
                st.info("No time-based data available for trend analysis.")
        
        # Customer Metrics Tab
        with visualization_tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                # CLV vs Churn Probability
                if 'Churn_Probability' in clv_results_filtered.columns:
                    fig_churn = px.scatter(
                        clv_results_filtered,
                        x=clv_column,
                        y='Churn_Probability',
                        color='Customer_Type' if 'Customer_Type' in clv_results_filtered.columns else None,
                        title='CLV vs Churn Probability',
                        labels={
                            clv_column: 'Customer Lifetime Value ($)', 
                            'Churn_Probability': 'Churn Probability'
                        }
                    )
                    st.plotly_chart(fig_churn, use_container_width=True)
                else:
                    st.info("Churn probability data not available.")
            
            with col2:
                # CLV vs Customer Acquisition Cost
                if 'Customer_Acquisition_Cost' in clv_results_filtered.columns:
                    fig_cac = px.scatter(
                        clv_results_filtered,
                        x=clv_column,
                        y='Customer_Acquisition_Cost',
                        color='Acquisition_Channel' if 'Acquisition_Channel' in clv_results_filtered.columns else None,
                        title='CLV vs Customer Acquisition Cost',
                        labels={
                            clv_column: 'Customer Lifetime Value ($)', 
                            'Customer_Acquisition_Cost': 'Acquisition Cost ($)'
                        }
                    )
                    st.plotly_chart(fig_cac, use_container_width=True)
                else:
                    st.info("Customer acquisition cost data not available.")
        
        # Revenue Impact Tab
        with visualization_tabs[3]:
            # Segment Revenue Contribution
            segment_column = 'Value_Tier' if 'Value_Tier' in clv_results_filtered.columns else 'CLV_Segment'
            
            if segment_column in clv_results_filtered.columns:
                # Revenue Contribution by Segment
                segment_revenue = clv_results_filtered.groupby(segment_column)[clv_column].sum().reset_index()
                
                fig_revenue_contrib = px.pie(
                    segment_revenue,
                    values=clv_column,
                    names=segment_column,
                    title='Revenue Contribution by Customer Segment',
                    hole=0.3
                )
                st.plotly_chart(fig_revenue_contrib, use_container_width=True)
                
                # Retention Rate by Segment
                if 'Retention_Rate' in clv_results_filtered.columns:
                    retention_by_segment = clv_results_filtered.groupby(segment_column)['Retention_Rate'].mean().reset_index()
                    
                    fig_retention = px.bar(
                        retention_by_segment,
                        x=segment_column,
                        y='Retention_Rate',
                        title='Retention Rate by Customer Segment',
                        labels={'Retention_Rate': 'Average Retention Rate'}
                    )
                    st.plotly_chart(fig_retention, use_container_width=True)
                else:
                    st.info("Retention rate data not available.")
            else:
                st.info("No customer segmentation data available.")
        
        # RFM Analysis Tab
        with visualization_tabs[4]:
            # Perform RFM Analysis
            rfm_results = self._generate_rfm_analysis(clv_results_filtered)
            
            if rfm_results is not None:
                # RFM Segment Distribution
                segment_distribution = rfm_results['RFM_Segment'].value_counts()
                
                fig_rfm_segments = px.pie(
                    values=segment_distribution.values,
                    names=segment_distribution.index,
                    title='Customer Segments by RFM Analysis',
                    hole=0.3
                )
                st.plotly_chart(fig_rfm_segments, use_container_width=True)
                
                # RFM Scatter Plot
                fig_rfm_scatter = px.scatter(
                    rfm_results,
                    x='Frequency_Score',
                    y='Monetary_Score',
                    color='RFM_Segment',
                    size='RFM_Score',
                    hover_data=[clv_column],
                    title='RFM Segmentation',
                    labels={
                        'Frequency_Score': 'Frequency Score', 
                        'Monetary_Score': 'Monetary Score'
                    }
                )
                st.plotly_chart(fig_rfm_scatter, use_container_width=True)
            else:
                st.warning("Insufficient data for RFM analysis. Please ensure you have columns for purchase date, number of purchases, and total spend.")
    # Add these methods to the CLVAnalysisPage class

    def _check_for_clv_risk(self, clv_results):
        """
        Check for high risk conditions in CLV analysis and save to session_state
        
        Args:
            clv_results (pd.DataFrame): CLV analysis results
        
        Returns:
            bool: True if high risk is detected, False otherwise
        """
        try:
            # Define risk thresholds
            low_clv_risk_threshold = 0.3  # Consider it risky if 30% of customers have very low CLV
            
            # Find the CLV column to use
            clv_column_priority = ['Predicted_CLV', 'CLV', 'CLV_Adjusted', 'Discounted_CLV']
            clv_column = next((col for col in clv_column_priority if col in clv_results.columns), None)
            
            if clv_column is None:
                # Can't detect risk without a CLV column
                st.session_state['high_clv_risk'] = False
                return False
            
            # Calculate percentile thresholds
            low_clv_threshold = clv_results[clv_column].quantile(0.2)  # Bottom 20% threshold
            
            # Calculate metrics
            pct_low_clv = (clv_results[clv_column] < low_clv_threshold).mean()
            avg_clv = clv_results[clv_column].mean()
            
            # Check if Value_Tier column exists (Premium, High, Medium, Low)
            if 'Value_Tier' in clv_results.columns:
                # Check distribution of value tiers
                value_tier_counts = clv_results['Value_Tier'].value_counts(normalize=True)
                
                # Check if low-value customers are disproportionately high
                if 'Low' in value_tier_counts and value_tier_counts['Low'] > 0.4:  # More than 40% low-value customers
                    st.session_state['high_clv_risk'] = True
                    st.session_state['clv_risk_reason'] = f"High proportion of low-value customers: {value_tier_counts['Low']*100:.1f}%"
                    st.session_state['avg_clv'] = avg_clv
                    
                    # Store more metrics
                    st.session_state['clv'] = avg_clv
                    st.session_state['low_value_percentage'] = value_tier_counts['Low'] * 100
                    
                    st.warning(f"⚠️ CLV Risk: High proportion of low-value customers ({value_tier_counts['Low']*100:.1f}%)")
                    return True
            
            # Check for general low CLV risk
            if pct_low_clv > low_clv_risk_threshold:
                st.session_state['high_clv_risk'] = True
                st.session_state['clv_risk_reason'] = f"Large portion of customer base ({pct_low_clv*100:.1f}%) has very low CLV"
                st.session_state['avg_clv'] = avg_clv
                
                # Store more metrics
                st.session_state['clv'] = avg_clv
                st.session_state['low_clv_percentage'] = pct_low_clv * 100
                
                st.warning(f"⚠️ CLV Risk: {pct_low_clv*100:.1f}% of customers have very low lifetime value")
                return True
            
            # Check Pareto principle - if top 20% customers contribute more than 80% of value
            if 'Is_Top_Percentile' in clv_results.columns:
                top_customers = clv_results[clv_results['Is_Top_Percentile']]
                top_clv_contribution = top_customers[clv_column].sum() / clv_results[clv_column].sum()
                
                if top_clv_contribution > 0.8:  # 80/20 rule extreme
                    st.session_state['high_clv_risk'] = True
                    st.session_state['clv_risk_reason'] = f"Top {len(top_customers)/len(clv_results)*100:.1f}% of customers contribute {top_clv_contribution*100:.1f}% of total CLV"
                    st.session_state['avg_clv'] = avg_clv
                    
                    # Store more metrics
                    st.session_state['clv'] = avg_clv
                    st.session_state['top_contribution_percentage'] = top_clv_contribution * 100
                    
                    st.warning(f"⚠️ CLV Risk: Business depends heavily on top {len(top_customers)/len(clv_results)*100:.1f}% of customers")
                    return True
            
            # No high risk detected
            st.session_state['high_clv_risk'] = False
            st.session_state['avg_clv'] = avg_clv
            
            # Store CLV in session state even if no risk is detected
            st.session_state['clv'] = avg_clv
            
            return False
            
        except Exception as e:
            print(f"Error in CLV risk detection: {e}")
            st.session_state['high_clv_risk'] = False
            return False

    def _send_clv_risk_alert(self):
        """
        Send email alert for high CLV risk
        """
        try:
            # Check if user email exists in session state
            if 'user_email' not in st.session_state:
                # Fallback to get user info from authentication
                user_info = get_user_info()
                if user_info and 'email' in user_info:
                    st.session_state['user_email'] = user_info['email']
                else:
                    st.warning("User email not found. Cannot send alert.")
                    return False
            
            # Get risk details
            risk_reason = st.session_state.get('clv_risk_reason', "Unspecified CLV risk")
            avg_clv = st.session_state.get('avg_clv', 0)
            
            # Create alert message
            subject = "⚠️ CLV Risk Detected"
            message = f"""
            Alert: Customer Lifetime Value risk detected!
            
            Risk Details:
            - {risk_reason}
            - Average Customer Lifetime Value: ${avg_clv:.2f}
            
            Please review your CLV analysis dashboard for detailed insights and recommended strategies.
            
            This is an automated alert from BizNexus AI.
            """
            
            # Send the alert
            from src.utils.notifications import send_alert_email
            sent = send_alert_email(
                to_email=st.session_state['user_email'],
                subject=subject,
                message=message
            )
            
            if sent:
                st.success("CLV risk alert sent to your email.")
            else:
                st.error("Failed to send CLV risk alert email.")
            
            return sent
        
        except Exception as e:
            st.error(f"Error sending CLV risk alert: {str(e)}")
            return False  
                
    def render(self):
        """
        Main rendering method for CLV Analysis page
        """
        st.title("💎 Customer Lifetime Value (CLV) Analysis")
        
        # Initialize session state variables if they don't exist
        if 'run_clv_analysis' not in st.session_state:
            st.session_state.run_clv_analysis = False
        if 'clv_results' not in st.session_state:
            st.session_state.clv_results = None
        if 'clv_descriptive_stats' not in st.session_state:
            st.session_state.clv_descriptive_stats = None
        if 'clv_model' not in st.session_state:
            st.session_state.clv_model = None
                
        # Validate data availability
        if not self._validate_data_availability():
            return
        
        # Get customer dataframe from session state
        customer_df = st.session_state.customer_df
        
        # CLV Configuration
        clv_config = self._clv_configuration_options()
        
        # Perform Analysis Button
        run_analysis = st.button("Run CLV Analysis", type="primary")
        
        if run_analysis:
            # Store the button state in session to ensure it persists after rerun
            st.session_state.run_clv_analysis = True
        
        # Check if analysis should be run (either from button or session state)
        if st.session_state.get('run_clv_analysis', False):
            # Perform CLV analysis
            clv_analysis = self._perform_clv_analysis(customer_df, clv_config)
            
            if clv_analysis['success']:
                # Store results in session state for downloading
                st.session_state.clv_results = clv_analysis['clv_results']
                st.session_state.clv_descriptive_stats = clv_analysis['descriptive_stats']
                
                # Store the model if available
                if 'model' in clv_analysis:
                    st.session_state.clv_model = clv_analysis['model']
                
                # Show dataset preview
                self._show_dataset_preview(clv_analysis['clv_results'])
                
                # Descriptive Analysis (passing the model if available)
                # Descriptive Analysis
                self._descriptive_analysis(
                    clv_analysis['clv_results'], 
                    clv_analysis['descriptive_stats']
                )
                
                # Business Insights
                insights_config = self._key_insights_and_visualization(clv_analysis['clv_results'], clv_config)
                
                # Option to proceed to next stage (Churn Prediction)
                col1, col2, col3 = st.columns([1,2,1])
                
                with col2:
                    if st.button("Continue to Churn Prediction", type="primary", use_container_width=True):
                        try:
                            # Attempt navigation using Streamlit's built-in navigation
                            st.switch_page("pages/05_Churn_Prediction.py")
                        except Exception as e:
                            st.error(f"Navigation error: {e}")
                            
                            # Fallback navigation method
                            try:
                                from src.auth.session import nav_to
                                nav_to("05_Churn_Prediction")
                            except Exception as nav_error:
                                st.error(f"Fallback navigation failed: {nav_error}")
                                st.warning("Please manually navigate to the Churn Prediction page.")
            
@requires_auth
def main():
    """
    Main function to initialize and render the CLV Analysis page
    """
    # Create an instance of the CLVAnalysisPage class
    clv_page = CLVAnalysisPage()
    
    # Render the page
    clv_page.render()

# Ensure the script can be run directly
if __name__ == "__main__":
    main()