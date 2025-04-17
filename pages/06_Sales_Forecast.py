# Core Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime, timedelta

# Notification and Authentication Imports

from src.auth.session import requires_auth, get_user_info, get_company_id
from src.auth.firebase_auth import retrieve_user_smtp_config, encrypt_sensitive_data, decrypt_sensitive_data

# Reporting and Model Imports
from src.reports.sales_report_generator import download_report
from src.models.sales_forecasting_model import SalesForecastingModel

# PDF and Excel Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

class AdvancedSalesForecastingDashboard:
    def __init__(self):
        # Page configuration
        st.set_page_config(
            page_title="AI Sales Forecasting Dashboard", 
            page_icon="📈", 
            layout="wide"
        )
        
        # Apply custom styling
        self._apply_custom_styling()
        
        # Initialize session state for configuration
        self._initialize_session_state()
    
    def _apply_custom_styling(self):
        """Apply professional dashboard styling"""
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
        .dashboard-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .dashboard-card:hover {
            transform: scale(1.02);
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
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: 600;
            color: white;
            background-color: #4a90e2;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3a80d2;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        button[data-testid="baseButton-secondary"] {
            background-color: #4a90e2;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _initialize_session_state(self):
        """Initialize session state for forecasting configuration"""
        if 'forecast_config' not in st.session_state:
            st.session_state.forecast_config = {
                'forecast_horizon': 90,
                'confidence_interval': 95,
                'data_frequency': 'Monthly',
                'seasonality': True,
                'historical_period': '1 Year',
                'external_variables': [],
                'filter_by': []
            }
        
        # Initialize forecast data in session state
        if 'forecast_data' not in st.session_state:
            st.session_state.forecast_data = None
        
        # Initialize page navigation
        if 'page' not in st.session_state:
            st.session_state.page = "sales_forecast"
    
    def _configuration_panel(self):
        """
        Create comprehensive configuration panel for sales forecasting
        """
        st.markdown("## 🔧 Sales Forecasting Configuration")
        
        config = st.session_state.forecast_config
        
        # Configuration Columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📅 Forecast Horizon")
            config['forecast_horizon'] = st.selectbox(
                "Forecast Period", 
                [30, 60, 90, 180, 365],
                index=2,
                help="Number of days to forecast sales"
            )
        
        with col2:
            st.markdown("### 📊 Data Granularity")
            config['data_frequency'] = st.selectbox(
                "Aggregation Frequency", 
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                help="Level of detail for sales forecast"
            )
        
        with col3:
            st.markdown("### 🎯 Confidence Interval")
            config['confidence_interval'] = st.selectbox(
                "Prediction Interval", 
                [80, 90, 95, 99],
                help="Statistical confidence for forecast prediction"
            )
        
        # Advanced Configuration Expander
        with st.expander("🚀 Advanced Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Historical Data")
                config['historical_period'] = st.selectbox(
                    "Training Data Period", 
                    ["6 Months", "1 Year", "3 Years"],
                    help="Historical data used for training forecast model"
                )
                
                st.markdown("### 🔍 Seasonality")
                config['seasonality'] = st.checkbox(
                    "Enable Seasonality Detection", 
                    value=True,
                    help="Automatically detect and adjust for seasonal patterns"
                )
            
            with col2:
                st.markdown("### 🌐 External Variables")
                external_vars = [
                    "Promotions", 
                    "Holidays", 
                    "Economic Indicators", 
                    "Weather", 
                    "Competitor Pricing"
                ]
                config['external_variables'] = st.multiselect(
                    "Select External Factors", 
                    external_vars,
                    help="Additional variables to consider in forecasting"
                )
        
        # Filtering and Segmentation
        st.markdown("## 🔬 Forecast Segmentation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📦 Product Segmentation")
            config['filter_by'] = st.multiselect(
                "Filter Forecast By", 
                ["Product Category", "SKU", "Region", "Sales Channel"],
                help="Segment your sales forecast"
            )
        
        # Update session state
        st.session_state.forecast_config = config
        
        return config
    
    def _validate_data_availability(self):
        """
        Validate that required data is available in session state
        
        Returns:
            bool: True if data is available, False otherwise
        """
        required_keys = ['customer_df', 'result_df']
        
        for key in required_keys:
            if key not in st.session_state:
                st.error(f"Missing {key}. Please complete previous steps.")
                return False
        
        return True

    def _generate_forecast_from_real_data(self, config):
        """
        Generate forecast using real customer data instead of mock data
        
        Args:
            config (dict): Forecasting configuration
        
        Returns:
            pd.DataFrame: Forecast data
        """
        try:
            # Get actual data from session state
            if 'customer_df' not in st.session_state or 'result_df' not in st.session_state:
                st.error("Missing required data. Please complete previous steps.")
                return None
            
            # Get customer and transaction data
            customer_df = st.session_state.customer_df
            result_df = st.session_state.result_df
            
            # Check if we have CLV and churn results from previous steps
            clv_results = st.session_state.get('clv_results')
            churn_results = st.session_state.get('churn_results', {}).get('churn_predictions')
            
            st.info("Using your actual customer and transaction data for forecasting")
            
            # Initialize the forecasting model with real data
            forecasting_model = SalesForecastingModel(
                customer_data=customer_df,
                transaction_data=result_df,
                clv_data=clv_results,
                churn_data=churn_results,
                config=config
            )
            
            # Generate the sales forecast
            forecast_results = forecasting_model.generate_forecast()
            
            # Use the forecast model results
            if isinstance(forecast_results, pd.DataFrame) and not forecast_results.empty:
                # Save to session state
                st.session_state.forecast_data = forecast_results
                
                # Check for sales risk
                is_high_risk = self._check_for_sales_risk(forecast_results)
                
                # If high risk is detected, send an alert
                if is_high_risk:
                    self._send_sales_risk_alert()
                
                return forecast_results
            else:
                # Fallback to mock data if real forecast failed
                st.warning("Could not generate forecast from your data. Using simulated data instead.")
                return self._generate_mock_forecast_data(config)
            # Store in session state
            
                    
        except Exception as e:
            st.error(f"Error generating forecast from real data: {str(e)}")
            st.warning("Falling back to simulated data.")
            # Fallback to mock data
            return self._generate_mock_forecast_data(config)

    def _check_for_sales_risk(self, forecast_data):
        """
        Check for high risk conditions in sales forecast and save to session_state
        
        Args:
            forecast_data (pd.DataFrame): Sales forecast data
        
        Returns:
            bool: True if high risk is detected, False otherwise
        """
        try:
            # Calculate key metrics
            # Looks at negative growth rate or dramatic forecast decrease
            
            # Calculate the growth trajectory (slope of forecast)
            first_forecast = forecast_data['Forecast'].iloc[0]
            last_forecast = forecast_data['Forecast'].iloc[-1]
            growth_rate = (last_forecast / first_forecast - 1) * 100
            
            # Store sales forecast in session state
            st.session_state['sales_forecast'] = forecast_data['Forecast'].mean()
            
            # Check for declining sales (negative growth rate)
            if growth_rate < -10:  # If sales declining by more than 10%
                st.session_state['high_sales_risk'] = True
                st.session_state['sales_growth_rate'] = growth_rate
                st.warning(f"⚠️ High Sales Risk: Projected decline of {growth_rate:.1f}%")
                return True
            
            # Check for high volatility (using coefficient of variation)
            forecast_mean = forecast_data['Forecast'].mean()
            forecast_std = forecast_data['Forecast'].std()
            forecast_cv = forecast_std / forecast_mean if forecast_mean > 0 else 0
            
            if forecast_cv > 0.3:  # If coefficient of variation is high
                st.session_state['high_sales_risk'] = True
                st.session_state['sales_volatility'] = forecast_cv
                st.warning(f"⚠️ High Sales Risk: High volatility detected ({forecast_cv:.2f})")
                return True
            
            # No high risk detected
            st.session_state['high_sales_risk'] = False
            st.session_state['sales_growth_rate'] = growth_rate
            st.session_state['sales_volatility'] = forecast_cv
            return False
        
        except Exception as e:
            print(f"Error in sales risk detection: {e}")
            st.session_state['high_sales_risk'] = False
            return False

    def _send_sales_risk_alert(self):
        """
        Send sales risk alert to the user
        """
        try:
            # Get user info from Firebase authentication
            user_info = get_user_info()
            
            if not user_info:
                st.error("❌ Unable to retrieve user information.")
                return False
            
            # Retrieve user details
            user_email = user_info.get('email')
            user_name = user_info.get('full_name', 'Valued Customer')
            company_name = user_info.get('company_name', 'Your Company')
            
            # Get risk details from session state
            sales_growth_rate = st.session_state.get('sales_growth_rate', 0)
            sales_volatility = st.session_state.get('sales_volatility', 0)
            sales_forecast = st.session_state.get('sales_forecast', 0)
            
            # Prepare alert message
            subject = f"⚠️ Sales Risk Alert - {company_name}"
            message = f"""
            Dear {user_name},

            Sales Risk Analysis for {company_name}
            
            Risk Details:
            - Projected Sales Growth: {sales_growth_rate:.1f}%
            - Sales Volatility: {sales_volatility:.2f}
            - Average Sales Forecast: ${sales_forecast:,.2f}

            Recommended Actions:
            1. Review sales strategies for declining product lines
            2. Analyze market conditions affecting sales volatility
            3. Develop targeted campaigns for growth opportunities

            Log into BizNexus AI for comprehensive insights.

            Best regards,
            BizNexus AI
            """
            
            # Send alert using the simplified function
            from src.utils.notifications import send_alert_email
            
            sent = send_alert_email(
                to_email=user_email,
                subject=subject,
                message=message
            )
            
            if sent:
                st.toast("✉️ Sales Risk Alert Sent!", icon="🚨")
                st.success(f"🔔 Alert sent to {user_email}")
            else:
                st.error("❌ Failed to send sales risk alert")
            
            return sent
        
        except Exception as e:
            st.error(f"❌ Alert sending error: {str(e)}")
            return False
    def _sales_trend_visualization(self, forecast_data):
        """
        Create comprehensive sales trend visualizations
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
        """
        st.markdown("## 📊 Sales Forecast Visualizations")
        
        # Tabs for different visualizations
        tabs = st.tabs([
            "Sales Forecast Trend", 
            "Seasonality & Anomalies", 
            "Promotional Impact Simulation"
        ])
        
        # Sales Forecast Trend Tab
        with tabs[0]:
            # Line chart with confidence interval
            fig_forecast = go.Figure([
                go.Scatter(
                    x=forecast_data['Date'], 
                    y=forecast_data['Forecast'], 
                    mode='lines', 
                    name='Forecast',
                    line=dict(color='blue', width=3)
                ),
                go.Scatter(
                    x=forecast_data['Date'].tolist() + forecast_data['Date'].tolist()[::-1],
                    y=forecast_data['Upper_Bound'].tolist() + forecast_data['Lower_Bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(68, 168, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    hoverinfo='skip'
                )
            ])
            fig_forecast.update_layout(
                title='Sales Forecast with Confidence Interval',
                xaxis_title='Date',
                yaxis_title='Forecasted Sales ($)'
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Seasonality & Anomalies Tab
        with tabs[1]:
            # Moving Average with Anomaly Detection
            window_size = 5  # Adjust based on data frequency
            forecast_data['Moving_Average'] = forecast_data['Forecast'].rolling(window=window_size).mean()
            forecast_data['Std_Dev'] = forecast_data['Forecast'].rolling(window=window_size).std()
            
            # Identify anomalies (beyond 2 standard deviations)
            forecast_data['Anomaly'] = np.abs(forecast_data['Forecast'] - forecast_data['Moving_Average']) > (2 * forecast_data['Std_Dev'])
            
            fig_anomalies = go.Figure()
            fig_anomalies.add_trace(go.Scatter(
                x=forecast_data['Date'], 
                y=forecast_data['Forecast'], 
                mode='lines', 
                name='Actual Sales'
            ))
            fig_anomalies.add_trace(go.Scatter(
                x=forecast_data['Date'][forecast_data['Anomaly']], 
                y=forecast_data['Forecast'][forecast_data['Anomaly']], 
                mode='markers', 
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            fig_anomalies.update_layout(
                title='Sales Trend with Anomaly Detection',
                xaxis_title='Date',
                yaxis_title='Sales'
            )
            st.plotly_chart(fig_anomalies, use_container_width=True)
        
        # Promotional Impact Simulation Tab
        with tabs[2]:
            # Simulate promotional impact
            promo_impact = forecast_data.copy()
            promo_dates = promo_impact['Date'].sample(n=3)
            promo_multipliers = [1.3, 1.5, 1.2]  # Sales boost during promotions
            
            for date, multiplier in zip(promo_dates, promo_multipliers):
                mask = promo_impact['Date'] == date
                promo_impact.loc[mask, 'Forecast'] *= multiplier
            
            fig_promo = go.Figure()
            fig_promo.add_trace(go.Scatter(
                x=forecast_data['Date'], 
                y=forecast_data['Forecast'], 
                mode='lines', 
                name='Base Sales'
            ))
            fig_promo.add_trace(go.Scatter(
                x=promo_impact['Date'], 
                y=promo_impact['Forecast'], 
                mode='lines', 
                name='Sales with Promotions',
                line=dict(color='green', dash='dot')
            ))
            
            # Highlight promotion dates
            for date, multiplier in zip(promo_dates, promo_multipliers):
                fig_promo.add_annotation(
                    x=date,
                    y=promo_impact.loc[promo_impact['Date'] == date, 'Forecast'].values[0],
                    text=f'Promo (+{(multiplier-1)*100:.0f}%)',
                    showarrow=True,
                    arrowhead=1
                )
            
            fig_promo.update_layout(
                title='Promotional Impact Simulation',
                xaxis_title='Date',
                yaxis_title='Sales'
            )
            st.plotly_chart(fig_promo, use_container_width=True)
    
    def _generate_kpi_metrics(self, forecast_data):
        """
        Generate and display key performance indicators
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
        """
        st.markdown("## 📈 Key Performance Indicators")
        
        # Prepare KPI calculations
        total_forecast = forecast_data['Forecast'].sum()
        avg_daily_sales = forecast_data['Forecast'].mean()
        growth_rate = (forecast_data['Forecast'].iloc[-1] / forecast_data['Forecast'].iloc[0] - 1) * 100
        
        # Create columns for KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">${:,.0f}</div>'.format(total_forecast), unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Sales Forecast</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">${:,.0f}</div>'.format(avg_daily_sales), unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Average Daily Sales</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">{:.1f}%</div>'.format(growth_rate), unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Sales Growth Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">{:.1f}%</div>'.format(95), unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Forecast Accuracy</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Scenario Planning Section
        st.markdown("## 🎯 Scenario Planning")
        scenario_columns = st.columns(3)
        
        scenarios = [
            {
                'name': 'Baseline Scenario',
                'price_change': 0,
                'marketing_spend': 0,
                'discount_rate': 0
            },
            {
                'name': 'Aggressive Growth',
                'price_change': -10,
                'marketing_spend': 20,
                'discount_rate': 15
            },
            {
                'name': 'Conservative Strategy',
                'price_change': 5,
                'marketing_spend': -10,
                'discount_rate': 5
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            with scenario_columns[i]:
                st.markdown(f'### {scenario["name"]}')
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Price Change: {scenario["price_change"]}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Marketing Spend: {scenario["marketing_spend"]}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Discount Rate: {scenario["discount_rate"]}%</div>', unsafe_allow_html=True)
                
                # Simulate scenario impact
                scenario_impact = (1 + scenario['price_change']/100) * (1 + scenario['marketing_spend']/100) * (1 - scenario['discount_rate']/100)
                st.markdown(f'<div class="metric-value">{scenario_impact*100-100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Estimated Sales Impact</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        
        # Actionable Insights Panel
        st.markdown("## 🤖 AI-Powered Recommendations")
        
        # Simulate AI recommendations
        recommendations = [
            "🚀 Increase marketing spend by 15% for high-performing product categories",
            "💡 Implement targeted discounts for products with declining sales",
            "🔍 Optimize inventory for top-selling items to prevent stockouts",
            "📊 Consider expanding into emerging market segments"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    def _report_export_section(self, forecast_data):
        """
        Create export and reporting section
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
        """
        st.markdown("## 📤 Export & Reporting")
        col1, col2 = st.columns(2)
        
        # Prepare KPI metrics dictionary
        kpi_metrics_dict = {
            'total_forecast': forecast_data['Forecast'].sum(),
            'avg_daily_sales': forecast_data['Forecast'].mean(),
            'growth_rate': (forecast_data['Forecast'].iloc[-1] / forecast_data['Forecast'].iloc[0] - 1) * 100
        }
        
        # Customer segments data
        customer_segments = pd.DataFrame({
            'Segment': ['High-Value', 'Medium-Value', 'Low-Value', 'At-Risk'],
            'Count': [500, 1500, 2000, 1000],
            'Average_CLV': [5000, 2000, 500, 750],
            'Churn_Probability': [0.1, 0.3, 0.6, 0.8]
        })
        
        with col1:
            if st.button("Generate Sales Report (PDF)", key="pdf_button_main"):
                download_report(
                    forecast_data=forecast_data, 
                    kpi_metrics=kpi_metrics_dict, 
                    customer_segments=customer_segments, 
                    report_type='PDF',
                    unique_key='sales_forecast_pdf_main'
                )
                st.success("PDF report generated successfully!")
        
        with col2:
            if st.button("Export Forecast Data (Excel)", key="excel_button_main"):
                download_report(
                    forecast_data=forecast_data, 
                    kpi_metrics=kpi_metrics_dict, 
                    customer_segments=customer_segments, 
                    report_type='Excel',
                    unique_key='sales_forecast_excel_main'
                )
                st.success("Excel report generated successfully!")
        
        # Add Customer Lookup navigation button
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Continue to Customer Lookup →", key="customer_lookup_btn", 
                        help="Navigate to the Customer Lookup page"):
                # Instead of changing page in session state, use streamlit navigation
                st.switch_page("pages/08_Customer_Lookup.py")
    
    def _scenario_sensitivity_analysis(self, forecast_data):
        """
        Perform sensitivity analysis for different scenarios
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
        """
        st.markdown("## 🔬 Scenario Sensitivity Analysis")
        
        # Create interactive sliders for sensitivity analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price_change = st.slider(
                "Price Change (%)", 
                min_value=-30, 
                max_value=30, 
                value=0, 
                step=5
            )
        
        with col2:
            marketing_spend = st.slider(
                "Marketing Spend Change (%)", 
                min_value=-50, 
                max_value=50, 
                value=0, 
                step=5
            )
        
        with col3:
            discount_rate = st.slider(
                "Discount Rate (%)", 
                min_value=0, 
                max_value=30, 
                value=0, 
                step=5
            )
        
        # Simulate scenario impact
        sensitivity_data = forecast_data.copy()
        sensitivity_multiplier = (1 + price_change/100) * (1 + marketing_spend/100) * (1 - discount_rate/100)
        sensitivity_data['Forecast'] *= sensitivity_multiplier
        
        # Visualize sensitivity analysis
        fig_sensitivity = go.Figure()
        fig_sensitivity.add_trace(go.Scatter(
            x=forecast_data['Date'], 
            y=forecast_data['Forecast'], 
            mode='lines', 
            name='Base Forecast'
        ))
        fig_sensitivity.add_trace(go.Scatter(
            x=sensitivity_data['Date'], 
            y=sensitivity_data['Forecast'], 
            mode='lines', 
            name='Adjusted Forecast'
        ))
        
        fig_sensitivity.update_layout(
            title='Sales Forecast Sensitivity Analysis',
            xaxis_title='Date',
            yaxis_title='Forecasted Sales'
        )
        
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Impact summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Price Change Impact", 
                f"{price_change:.1f}%", 
                help="Percentage change in sales due to price adjustments"
            )
        
        with col2:
            st.metric(
                "Marketing Spend Impact", 
                f"{marketing_spend:.1f}%", 
                help="Percentage change in sales due to marketing spend"
            )
        
        with col3:
            st.metric(
                "Discount Rate Impact", 
                f"{discount_rate:.1f}%", 
                help="Percentage change in sales due to discounts"
            )
    
    def _customer_segmentation_analysis(self):
        """
        Perform customer segmentation analysis
        """
        st.markdown("## 👥 Customer Segmentation Analysis")
        
        # Simulate customer segmentation data
        np.random.seed(42)
        customer_segments = pd.DataFrame({
            'Segment': ['High-Value', 'Medium-Value', 'Low-Value', 'At-Risk'],
            'Count': [500, 1500, 2000, 1000],
            'Average_CLV': [5000, 2000, 500, 750],
            'Churn_Probability': [0.1, 0.3, 0.6, 0.8]
        })
        
        # Customer Segment Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of customer segments
            fig_segment_dist = px.pie(
                customer_segments, 
                values='Count', 
                names='Segment',
                title='Customer Segment Distribution',
                hole=0.3
            )
            st.plotly_chart(fig_segment_dist, use_container_width=True)
        
        with col2:
            # Bar chart of Customer Lifetime Value by Segment
            fig_clv = px.bar(
                customer_segments, 
                x='Segment', 
                y='Average_CLV',
                title='Average Customer Lifetime Value',
                color='Segment'
            )
            st.plotly_chart(fig_clv, use_container_width=True)
        
        # Detailed Segment Insights
        st.markdown("### 📋 Segment Insights")
        insights_cols = st.columns(len(customer_segments))
        
        for i, (_, segment) in enumerate(customer_segments.iterrows()):
            with insights_cols[i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{segment["Count"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{segment["Segment"]} Customers</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Avg CLV: ${segment["Average_CLV"]:,.0f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Churn Risk: {segment["Churn_Probability"]:.0%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    def render(self):
        """
        Main rendering method for Advanced Sales Forecasting Dashboard
        """
        # Default to sales forecast page
        st.title("🚀 AI-Powered Sales Forecasting Dashboard")
        
        # Configuration Panel
        config = self._configuration_panel()
        
        # Generate Forecast Data
        if st.button("Generate Forecast", type="primary"):
            # Generate mock forecast data based on configuration
            forecast_data = self._generate_mock_forecast_data(config)
            
            # Sales Trend Visualizations
            self._sales_trend_visualization(forecast_data)
            
            # KPI Metrics
            self._generate_kpi_metrics(forecast_data)
            
            # Customer Segmentation Analysis
            self._customer_segmentation_analysis()
            
            # Scenario Sensitivity Analysis
            self._scenario_sensitivity_analysis(forecast_data)
            
            # Export & Reporting Section (now at the end of the page)
            self._report_export_section(forecast_data)
        
        # If forecast data already exists in session state, display it
        elif 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
            forecast_data = st.session_state.forecast_data
            
            # Sales Trend Visualizations
            self._sales_trend_visualization(forecast_data)
            
            # KPI Metrics
            self._generate_kpi_metrics(forecast_data)
            
            # Customer Segmentation Analysis
            self._customer_segmentation_analysis()
            
            # Scenario Sensitivity Analysis
            self._scenario_sensitivity_analysis(forecast_data)
            
            # Export & Reporting Section (now at the end of the page)
            self._report_export_section(forecast_data)
    def _generate_mock_forecast_data(self, config):
        """
        Generate forecast data based on configuration
        
        Args:
            config (dict): Forecasting configuration
        
        Returns:
            pd.DataFrame: Forecast data
        """
        try:
            # Check if we have data in session state
            if 'customer_df' not in st.session_state or 'result_df' not in st.session_state:
                st.error("Missing required data. Please complete previous steps.")
                # Generate some mock data as a fallback
                return self._create_fallback_data(config)
            
            # Log what data we have in session state
            st.info("Using your uploaded data for forecasting")
            
            # Get customer and transaction data
            customer_df = st.session_state.customer_df
            result_df = st.session_state.result_df
            
            # Check if we have transaction or purchase data with dates
            if result_df is not None:
                # Look for date columns
                date_columns = [col for col in result_df.columns 
                            if any(date_term in col.lower() for date_term in ['date', 'time', 'day'])]
                
                if date_columns:
                    date_column = date_columns[0]
                    
                    # Look for amount/sales columns
                    amount_columns = [col for col in result_df.columns 
                                    if any(amount_term in col.lower() for amount_term in 
                                        ['amount', 'price', 'revenue', 'sales', 'value'])]
                    
                    if amount_columns:
                        amount_column = amount_columns[0]
                        
                        # Ensure date column is datetime
                        result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
                        
                        # Group by date and sum the amounts
                        sales_by_date = result_df.groupby(pd.Grouper(key=date_column, freq='D'))[amount_column].sum().reset_index()
                        
                        # We have historical sales data, now generate forecast
                        last_date = sales_by_date[date_column].max()
                        
                        # Generate future dates
                        forecast_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=config['forecast_horizon'],
                            freq='D'
                        )
                        
                        # Use simple forecasting method (you can replace with more sophisticated model)
                        # Calculate average and trend from recent data
                        recent_data = sales_by_date.tail(30)
                        avg_sales = recent_data[amount_column].mean()
                        
                        # Calculate trend
                        if len(recent_data) > 1:
                            x = np.arange(len(recent_data))
                            y = recent_data[amount_column].values
                            slope, intercept = np.polyfit(x, y, 1)
                            trend_factor = slope / avg_sales if avg_sales > 0 else 0.01
                        else:
                            trend_factor = 0.01
                        
                        # Generate forecast with trend and seasonality
                        trend = 1 + trend_factor * np.arange(len(forecast_dates))
                        seasonality = 1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(forecast_dates)))
                        noise = np.random.normal(0, 0.05, len(forecast_dates))
                        
                        forecast_values = avg_sales * trend * seasonality * (1 + noise)
                        
                        # Add confidence interval
                        std_dev = recent_data[amount_column].std()
                        ci_factor = 1.96  # 95% confidence interval
                        
                        lower_bound = forecast_values - ci_factor * std_dev
                        upper_bound = forecast_values + ci_factor * std_dev
                        
                        # Make sure lower bound is not negative
                        lower_bound = np.maximum(lower_bound, 0)
                        
                        # Create forecast DataFrame
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecast_values,
                            'Lower_Bound': lower_bound,
                            'Upper_Bound': upper_bound
                        })
                        
                        # Store in session state
                        st.session_state.forecast_data = forecast_df
                        
                        
                        
                        return forecast_df
            
            # If we couldn't generate forecast from real data, use fallback
            st.warning("Couldn't generate forecast from your data. Using simulated data.")
            return self._create_fallback_data(config)
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return self._create_fallback_data(config)

    def _create_fallback_data(self, config):
        """
        Create fallback forecast data when real data can't be used
        
        Args:
            config (dict): Forecasting configuration
        
        Returns:
            pd.DataFrame: Simulated forecast data
        """
        # Generate date range
        start_date = datetime.now()
        end_date = start_date + timedelta(days=config['forecast_horizon'])
        
        # Generate date index
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate sales data with seasonality and trend
        np.random.seed(42)
        base_sales = 10000  # Base daily sales
        trend = np.linspace(1, 1.2, len(date_range))  # Upward trend
        seasonality = np.sin(np.linspace(0, 4*np.pi, len(date_range))) * 0.2 + 1  # Seasonal component
        noise = np.random.normal(0, 0.1, len(date_range))
        
        sales = base_sales * trend * seasonality * (1 + noise)
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'Date': date_range,
            'Forecast': sales,
            'Lower_Bound': sales * (1 - config['confidence_interval']/100),
            'Upper_Bound': sales * (1 + config['confidence_interval']/100)
        })
        
        # Store in session state
        st.session_state.forecast_data = forecast_df
        
        # Apply risk detection
        if hasattr(self, '_check_for_sales_risk'):
            self._check_for_sales_risk(forecast_df)
        
        return forecast_df
    
    def render(self):
        """
        Main rendering method for Advanced Sales Forecasting Dashboard
        """
        # Default to sales forecast page
        st.title("🚀 AI-Powered Sales Forecasting Dashboard")
        
        # First validate if we have data
        if hasattr(self, '_validate_data_availability') and not self._validate_data_availability():
            st.error("Please complete previous steps to upload and process your data.")
            
            # Add navigation button to previous page
            if st.button("Go to Data Upload", type="primary"):
                try:
                    st.switch_page("pages/03_Data_Transformation.py")
                except Exception as e:
                    st.error(f"Navigation error: {e}")
                    
            return
        
        # Configuration Panel
        config = self._configuration_panel()
        
        # Generate Forecast Data
        if st.button("Generate Forecast", type="primary"):
            # Generate forecast using real data from session state
            forecast_data = self._generate_mock_forecast_data(config)
            
            if forecast_data is not None:
                # Sales Trend Visualizations
                self._sales_trend_visualization(forecast_data)
                
                # KPI Metrics
                self._generate_kpi_metrics(forecast_data)
                
                # Customer Segmentation Analysis
                self._customer_segmentation_analysis()
                
                # Scenario Sensitivity Analysis
                self._scenario_sensitivity_analysis(forecast_data)
                
                # Export & Reporting Section (now at the end of the page)
                self._report_export_section(forecast_data)
        
        # If forecast data already exists in session state, display it
        elif 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
            forecast_data = st.session_state.forecast_data
            
            # Sales Trend Visualizations
            self._sales_trend_visualization(forecast_data)
            
            # KPI Metrics
            self._generate_kpi_metrics(forecast_data)
            
            # Customer Segmentation Analysis
            self._customer_segmentation_analysis()
            
            # Scenario Sensitivity Analysis
            self._scenario_sensitivity_analysis(forecast_data)
            
            # Export & Reporting Section (now at the end of the page)
            self._report_export_section(forecast_data)
        # Add Alert navigation buttons in sidebar
        elif st.session_state.authenticated:
            # Create a sidebar section for Alerts
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Alerts")
            
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.sidebar.button("Alert Settings", key="nav_alert_settings"):
                    try:
                        st.switch_page("pages/10_Alert_Settings.py")
                    except Exception as e:
                        st.error(f"Navigation error: {str(e)}")
            
            with col2:
                if st.sidebar.button("Alert History", key="nav_alert_history"):
                    try:
                        st.switch_page("pages/11_Alert_History.py")
                    except Exception as e:
                        st.error(f"Navigation error: {str(e)}")
        else:
            st.error("Please complete previous steps to upload and process your data.")
            
            # Add navigation button to previous page
            if st.button("Go to Data Upload", type="primary"):
                try:
                    st.switch_page("pages/03_Data_Transformation.py")
                except Exception as e:
                    st.error(f"Navigation error: {e}")
            
        
# Add main function with authentication decorator
@requires_auth
def main():
    """
    Main function to initialize and render the Advanced Sales Forecasting Dashboard
    """
    # Create an instance of the dashboard
    dashboard = AdvancedSalesForecastingDashboard()
    
    # Handle navigation between pages
    if 'page' not in st.session_state:
        st.session_state.page = "sales_forecast"
    
    # Render the dashboard based on current page
    dashboard.render()

# Ensure the script can be run directly
if __name__ == "__main__":
    main()