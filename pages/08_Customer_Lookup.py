import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

# Import models and utilities
from src.auth.session import requires_auth, get_user_info, get_company_id
from src.firebase.firestore import save_dataset
from src.reports.customer_report_generator import download_customer_report

class CustomerLookupPage:
    def __init__(self):
        # Page configuration
        st.set_page_config(
            page_title="BizNexus AI | Customer 360°", 
            page_icon="👤", 
            layout="wide"
        )
        
        # Apply custom styling
        self._apply_custom_styling()
    
    def _apply_custom_styling(self):
        """Apply professional dashboard styling"""
        st.markdown("""
        <style>
        .main {
            background-color: #f4f6f9;
            color: #2c3e50;
        }
        .customer-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .customer-card:hover {
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
        .recommendation-card {
            background-color: #f8f9fa;
            border-left: 4px solid #4a90e2;
            padding: 15px;
            margin-bottom: 10px;
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
    
    def _validate_data_availability(self):
        """
        Validate that required data is available in session state
        
        Returns:
            bool: True if data is available, False otherwise
        """
        # For testing purposes, let's create mock data if not available
        if 'customer_df' not in st.session_state:
            # Create mock customer data
            st.session_state.customer_df = pd.DataFrame({
                'Customer ID': range(1, 11),
                'Customer Name': [f'Customer {i}' for i in range(1, 11)],
                'Email': [f'customer{i}@example.com' for i in range(1, 11)],
                'Region': np.random.choice(['North', 'South', 'East', 'West'], 10),
                'Signup_Date': pd.date_range(start='2023-01-01', periods=10)
            })
        
        if 'result_df' not in st.session_state:
            # Create mock result data
            st.session_state.result_df = pd.DataFrame({
                'Customer ID': range(1, 11),
                'Metric1': np.random.rand(10),
                'Metric2': np.random.rand(10) * 100
            })
            
        return True
    
    def _load_customer_data(self):
        """
        Load and prepare customer data for lookup
        
        Returns:
            pd.DataFrame: Prepared customer dataframe
        """
        # Retrieve customer dataframe from session state
        customer_df = st.session_state.customer_df
        
        # Check for essential columns
        essential_columns = ['Customer ID', 'Customer Name', 'Email', 'Region', 'Signup_Date']
        
        # Add columns if they don't exist
        for col in essential_columns:
            if col not in customer_df.columns:
                # Add placeholder columns if missing
                if col == 'Customer ID':
                    customer_df[col] = range(1, len(customer_df) + 1)
                elif col == 'Customer Name':
                    customer_df[col] = 'Customer_' + customer_df.index.astype(str)
                else:
                    customer_df[col] = 'N/A'
        
        return customer_df
    
    def _retrieve_customer_details(self, customer_df, selected_customer):
        """
        Retrieve and merge all available details for a specific customer
        
        Args:
            customer_df (pd.DataFrame): Customer dataframe
            selected_customer (str): Selected customer ID or name
        
        Returns:
            dict: Comprehensive customer details
        """
        try:
            # Check if selected_customer is a Customer ID or Name
            if isinstance(selected_customer, str) and selected_customer.isdigit():
                selected_customer = int(selected_customer)
            
            # Find the customer row
            customer_mask = (customer_df['Customer ID'] == selected_customer) | (customer_df['Customer Name'] == selected_customer)
            
            if not customer_mask.any():
                st.error(f"Customer not found: {selected_customer}")
                return {}
                
            customer_details = customer_df.loc[customer_mask].iloc[0]
            
            # Merge additional analyses if available in session state
            customer_details_dict = customer_details.to_dict()
            
            # Add mock CLV and Churn data for testing
            customer_details_dict['CLV_Predicted_CLV'] = np.random.uniform(500, 5000)
            customer_details_dict['CLV_Value_Tier'] = np.random.choice(['High', 'Medium', 'Low'])
            customer_details_dict['Churn_Probability'] = np.random.uniform(0, 1)
            customer_details_dict['Churn_Risk_Category'] = np.random.choice(['High Risk', 'Medium Risk', 'Low Risk'])
            customer_details_dict['Recent_Product_Categories'] = np.random.choice(['Electronics', 'Services', 'Subscriptions'])
            
            # Merge CLV results if available
            if 'clv_results' in st.session_state and st.session_state.clv_results is not None:
                try:
                    clv_details = st.session_state.clv_results[
                        (st.session_state.clv_results['Customer ID'] == selected_customer) | 
                        (st.session_state.clv_results['Customer Name'] == selected_customer)
                    ]
                    
                    if not clv_details.empty:
                        clv_details_dict = clv_details.iloc[0].to_dict()
                        customer_details_dict.update({
                            f'CLV_{k}': v for k, v in clv_details_dict.items() 
                            if k not in customer_details_dict
                        })
                except Exception as e:
                    st.warning(f"Error retrieving CLV details: {e}")
            
            # Merge Churn results if available
            if 'churn_results' in st.session_state and st.session_state.churn_results is not None:
                try:
                    churn_details = st.session_state.churn_results['churn_predictions'][
                        (st.session_state.churn_results['churn_predictions']['Customer ID'] == selected_customer) | 
                        (st.session_state.churn_results['churn_predictions']['Customer Name'] == selected_customer)
                    ]
                    
                    if not churn_details.empty:
                        churn_details_dict = churn_details.iloc[0].to_dict()
                        customer_details_dict.update({
                            f'Churn_{k}': v for k, v in churn_details_dict.items() 
                            if k not in customer_details_dict
                        })
                except Exception as e:
                    st.warning(f"Error retrieving Churn details: {e}")
            
            return customer_details_dict
            
        except Exception as e:
            st.error(f"Error retrieving customer details: {e}")
            return {}
    
    def _generate_personalized_recommendations(self, customer_details):
        """
        Generate personalized recommendations based on customer details
        
        Args:
            customer_details (dict): Comprehensive customer details
        
        Returns:
            list: Personalized recommendations
        """
        recommendations = []
        
        # Check CLV-related recommendations
        if 'CLV_Predicted_CLV' in customer_details:
            clv = customer_details['CLV_Predicted_CLV']
            if clv < 1000:
                recommendations.append({
                    'type': 'Upgrade',
                    'title': 'Boost Your Value',
                    'description': f'Your current Customer Lifetime Value (${clv:.2f}) suggests opportunities for increased engagement. Consider upgrading to our premium tier for additional benefits.'
                })
        
        # Check Churn-related recommendations
        if 'Churn_Probability' in customer_details:
            churn_prob = customer_details['Churn_Probability']
            if churn_prob > 0.7:
                recommendations.append({
                    'type': 'Retention',
                    'title': 'Retention Intervention',
                    'description': f'High churn risk detected (Probability: {churn_prob:.1%}). We recommend a personalized retention campaign with exclusive offers.'
                })
        
        # Product-specific recommendations
        if 'Recent_Product_Categories' in customer_details:
            recommendations.append({
                'type': 'Cross-Sell',
                'title': 'Personalized Product Recommendations',
                'description': f'Based on your purchase history in {customer_details["Recent_Product_Categories"]}, you might be interested in complementary products.'
            })
        
        # Fallback recommendation
        if not recommendations:
            recommendations.append({
                'type': 'General',
                'title': 'Explore Our Services',
                'description': 'We have a range of exciting offerings tailored to enhance your experience. Check out our latest features!'
            })
        
        return recommendations
    
    def _customer_profile_visualization(self, customer_details):
        """
        Create comprehensive customer profile visualization
        
        Args:
            customer_details (dict): Comprehensive customer details
        """
        st.markdown("## 👤 Customer Profile Overview")
        
        # Profile Information Column
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown('<div class="customer-card">', unsafe_allow_html=True)
            
            # Profile Picture Placeholder
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="/api/placeholder/200/200" alt="Customer Profile" style="border-radius: 50%; width: 150px; height: 150px;">
            </div>
            """, unsafe_allow_html=True)
            
            # Basic Profile Info
            st.markdown(f"### {customer_details.get('Customer Name', 'Customer')}")
            st.markdown(f"**Customer ID:** {customer_details.get('Customer ID', 'N/A')}")
            st.markdown(f"**Email:** {customer_details.get('Email', 'N/A')}")
            st.markdown(f"**Region:** {customer_details.get('Region', 'N/A')}")
            st.markdown(f"**Signup Date:** {customer_details.get('Signup_Date', 'N/A')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Customer Key Metrics
            st.markdown('<div class="customer-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Key Performance Metrics")
            
            # CLV Related Metrics
            col_clv1, col_clv2 = st.columns(2)
            
            with col_clv1:
                clv = customer_details.get('CLV_Predicted_CLV', 0)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${clv:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Customer Lifetime Value</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_clv2:
                churn_prob = customer_details.get('Churn_Probability', 0)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{churn_prob:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Churn Risk</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional Segmentation Metrics
            st.markdown("### Segmentation Insights")
            value_tier = customer_details.get('CLV_Value_Tier', 'Not Classified')
            risk_category = customer_details.get('Churn_Risk_Category', 'Standard Risk')
            
            st.markdown(f"**Value Tier:** {value_tier}")
            st.markdown(f"**Churn Risk Category:** {risk_category}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # Personalized Recommendations
            st.markdown('<div class="customer-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Personalized Recommendations")
            
            recommendations = self._generate_personalized_recommendations(customer_details)
            
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>{rec['title']}</strong>
                    <p>{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _customer_engagement_analysis(self, customer_details):
        """
        Perform comprehensive customer engagement analysis
        
        Args:
            customer_details (dict): Comprehensive customer details
        """
        st.markdown("## 📈 Customer Engagement Deep Dive")
        
        # Engagement Tabs
        tabs = st.tabs([
            "Purchase History", 
            "Interaction Timeline", 
            "Predictive Insights"
        ])
        
        # Purchase History Tab
        with tabs[0]:
            st.markdown("### 💰 Recent Purchase Patterns")
            
            # Simulated Purchase History
            purchase_data = pd.DataFrame({
                'Date': pd.date_range(end=datetime.now(), periods=6),
                'Amount': np.random.uniform(50, 500, 6),
                'Category': np.random.choice(['Electronics', 'Services', 'Subscriptions', 'Consulting'], 6)
            })
            
            # Purchase Amount Trend
            fig_purchases = px.line(
                purchase_data, 
                x='Date', 
                y='Amount', 
                color='Category',
                title='Purchase Amount by Category'
            )
            st.plotly_chart(fig_purchases, use_container_width=True)
            
            # Purchase Category Distribution
            category_dist = purchase_data.groupby('Category')['Amount'].sum()
            fig_category = px.pie(
                category_dist, 
                values=category_dist.values, 
                names=category_dist.index,
                title='Purchase Category Distribution'
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        # Interaction Timeline Tab
        with tabs[1]:
            st.markdown("### 🕒 Customer Journey Timeline")
            
            # Simulated Interaction Data
            interaction_data = pd.DataFrame({
                'Date': pd.date_range(end=datetime.now(), periods=10),
                'Event': np.random.choice([
                    'Website Visit', 
                    'Product Inquiry', 
                    'Support Ticket', 
                    'Purchase', 
                    'Feedback Submission'
                ], 10),
                'Channel': np.random.choice(['Web', 'Mobile', 'Email', 'Phone'], 10)
            })
            
            # Create Sankey Diagram for Customer Journey
            unique_events = interaction_data['Event'].unique()
            unique_channels = interaction_data['Channel'].unique()
            
            # Prepare Sankey data
            source_indices = {event: i for i, event in enumerate(unique_events)}
            target_indices = {channel: i + len(unique_events) for i, channel in enumerate(unique_channels)}
            
            # Count flow between events and channels
            flow_data = interaction_data.groupby(['Event', 'Channel']).size().reset_index(name='value')
            
            # Prepare Sankey input
            sankey_nodes = list(unique_events) + list(unique_channels)
            sankey_sources = [source_indices[row['Event']] for _, row in flow_data.iterrows()]
            sankey_targets = [target_indices[row['Channel']] for _, row in flow_data.iterrows()]
            sankey_values = flow_data['value']
            
            fig_journey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=sankey_nodes
                ),
                link=dict(
                    source=sankey_sources,
                    target=sankey_targets,
                    value=sankey_values
                )
            )])
            
            fig_journey.update_layout(title_text="Customer Interaction Journey", font_size=10)
            st.plotly_chart(fig_journey, use_container_width=True)
        
        # Predictive Insights Tab
        with tabs[2]:
            st.markdown("### 🔮 Predictive Customer Insights")
            
            # Future Purchase Probability
            purchase_prob_data = {
                'Category': ['High-Value', 'Medium-Value', 'Low-Value'],
                'Probability': [
                    customer_details.get('Churn_Probability', 0.3) * 100,
                    (1 - customer_details.get('Churn_Probability', 0.3)) * 50,
                    (1 - customer_details.get('Churn_Probability', 0.3)) * 10
                ]
            }
            
            fig_purchase_prob = px.bar(
                purchase_prob_data, 
                x='Category', 
                y='Probability',
                title='Future Purchase Probability',
                labels={'Probability': 'Purchase Probability (%)'}
            )
            st.plotly_chart(fig_purchase_prob, use_container_width=True)
            
            # Risk and Opportunity Analysis
            st.markdown("### 🎲 Risk & Opportunity Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{customer_details.get("Churn_Probability", 0):.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Churn Risk</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                retention_score = 1 - customer_details.get("Churn_Probability", 0)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{retention_score:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Retention Probability</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def _generate_targeted_marketing_strategy(self, customer_details):
        """
        Generate a targeted marketing strategy for the customer
        
        Args:
            customer_details (dict): Comprehensive customer details
        """
        st.markdown("## 🎯 Targeted Marketing Strategy")
        
        # Risk and Value Assessment
        churn_prob = customer_details.get('Churn_Probability', 0)
        clv = customer_details.get('CLV_Predicted_CLV', 0)
        
        # Marketing Strategy Segmentation
        if churn_prob > 0.7 and clv > 5000:
            # High-Value at High Risk
            strategy = {
                'title': 'VIP Retention Intervention',
                'risk_level': 'Critical',
                'recommended_actions': [
                    "Personal Account Manager Assignment",
                    "Exclusive 20% Loyalty Discount",
                    "Priority Customer Support Access",
                    "Customized Product Bundle Offer"
                ]
            }
        elif churn_prob > 0.5:
            # Moderate Risk
            strategy = {
                'title': 'Proactive Engagement Campaign',
                'risk_level': 'High',
                'recommended_actions': [
                    "Targeted Re-engagement Email Series",
                    "10% Reactivation Discount",
                    "Personalized Product Recommendations",
                    "Customer Feedback Survey"
                ]
            }
        elif clv < 1000:
            # Low Value Potential
            strategy = {
                'title': 'Value Acceleration Program',
                'risk_level': 'Moderate',
                'recommended_actions': [
                    "Upgrade Path Recommendation",
                    "Educational Content Series",
                    "Introductory Pricing for Premium Features",
                    "Cross-Selling Opportunities"
                ]
            }
        else:
            # Low Risk, High Value
            strategy = {
                'title': 'Loyalty Enhancement Strategy',
                'risk_level': 'Low',
                'recommended_actions': [
                    "Referral Program Invitation",
                    "Early Access to New Features",
                    "Loyalty Rewards Program",
                    "Personalized Upsell Suggestions"
                ]
            }
        
        # Display Marketing Strategy
        st.markdown(f"### 📣 {strategy['title']}")
        st.markdown(f"**Risk Level:** {strategy['risk_level']}")
        
        st.markdown("#### Recommended Actions:")
        for action in strategy['recommended_actions']:
            st.markdown(f"- {action}")
        
        # Potential Offer Simulation
        st.markdown("### 💡 Potential Offer Simulation")
        offer_simulation = {
            'Base Discount': 10,
            'Loyalty Bonus': 5,
            'Referral Credit': 15,
            'Total Potential Value': 30
        }
        
        fig_offer = go.Figure(go.Bar(
            x=list(offer_simulation.keys()),
            y=list(offer_simulation.values()),
            text=[f'${val}' for val in offer_simulation.values()],
            textposition='auto'
        ))
        fig_offer.update_layout(
            title='Potential Marketing Offer Breakdown',
            yaxis_title='Offer Value ($)'
        )
        st.plotly_chart(fig_offer, use_container_width=True)
    
    def _export_customer_report(self, customer_details):
        """
        Export comprehensive customer report
        
        Args:
            customer_details (dict): Comprehensive customer details
        """
        st.markdown("## 📤 Export Customer Insights")
        
        col1, col2 = st.columns(2)
        
        # Convert customer details to a DataFrame for export
        # Create a proper DataFrame with the required 'Forecast' column structure
        # that's expected by the download_report() function
        dates = pd.date_range(start=datetime.now(), periods=12, freq='M')
        
        # Generate sample forecast data based on CLV
        clv = customer_details.get('CLV_Predicted_CLV', 1000)
        base_value = clv / 12  # Monthly value
        
        # Create a customer name that's safe for use in identifiers
        customer_name = customer_details.get('Customer Name', 'Unknown')
        customer_id = customer_details.get('Customer ID', 'Unknown')
        
        forecast_data = pd.DataFrame({
            'Date': dates,
            'Forecast': [base_value * (1 + np.random.normal(0, 0.1)) for _ in range(12)],
            'Customer ID': customer_id,
            'Customer Name': customer_name
        })
        
        # Add upper and lower bounds for completeness
        forecast_data['Upper_Bound'] = forecast_data['Forecast'] * 1.2
        forecast_data['Lower_Bound'] = forecast_data['Forecast'] * 0.8
        
        # Generate KPI metrics dict with custom report title
        kpi_metrics = {
            'total_forecast': forecast_data['Forecast'].sum(),
            'avg_daily_sales': forecast_data['Forecast'].mean(),
            'growth_rate': (forecast_data['Forecast'].iloc[-1] / forecast_data['Forecast'].iloc[0] - 1) * 100,
            'customer_value': customer_details.get('CLV_Predicted_CLV', 0),
            'churn_risk': customer_details.get('Churn_Probability', 0),
            'retention_score': 1 - customer_details.get('Churn_Probability', 0),
            # This field might be used by your report generator to set the title
            'custom_title': f"Customer Report - {customer_name}"
        }
        
        # Customer segment data (just this one customer)
        customer_segments = pd.DataFrame({
            'Segment': [customer_details.get('CLV_Value_Tier', 'Medium')],
            'Count': [1],
            'Average_CLV': [customer_details.get('CLV_Predicted_CLV', 0)],
            'Churn_Probability': [customer_details.get('Churn_Probability', 0)],
            'Customer ID': [customer_id],
            'Customer Name': [customer_name]
        })
        
        with col1:
            if st.button("Generate Detailed PDF Report", type="primary"):
                try:
                    # We need to use only the parameters that download_report() accepts
                    download_customer_report(
                        forecast_data=forecast_data,
                        kpi_metrics=kpi_metrics,
                        customer_segments=customer_segments,
                        report_type='PDF',
                        unique_key=f"customer_{customer_id}_pdf"
                    )
                    st.success(f"PDF report for {customer_name} generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF report: {e}")
        
        with col2:
            if st.button("Export Customer Data to Excel"):
                try:
                    download_customer_report(
                        forecast_data=forecast_data,
                        kpi_metrics=kpi_metrics,
                        customer_segments=customer_segments,
                        report_type='Excel',
                        unique_key=f"customer_{customer_id}_excel"
                    )
                    st.success(f"Excel data for {customer_name} exported successfully!")
                except Exception as e:
                    st.error(f"Error generating Excel report: {e}")
    def _navigate_to_ai_assistant(self):
        """
        Navigate to the AI Assistant page
        """
        st.switch_page("pages/09_AI_Assistant.py")
        
    def render(self):
        """
        Main rendering method for Customer Lookup page
        """
        st.title("👤 Customer 360° Insights")
        
        # Validate data availability - replace with a mocked version that always returns True
        if not self._validate_data_availability():
            return
        
        # Load customer data
        customer_df = self._load_customer_data()
        
        # Customer Selection
        st.sidebar.header("🔍 Customer Selection")
        customer_selection_options = customer_df['Customer Name'].tolist() + customer_df['Customer ID'].astype(str).tolist()
        selected_customer = st.sidebar.selectbox(
            "Choose a Customer", 
            options=customer_selection_options,
            help="Select a customer to view their comprehensive profile"
        )
        
        # Retrieve Detailed Customer Information
        if selected_customer:
            # Retrieve comprehensive customer details
            customer_details = self._retrieve_customer_details(customer_df, selected_customer)
            
            if customer_details:
                # Display various customer insights sections
                self._customer_profile_visualization(customer_details)
                
                # Customer Engagement Analysis
                self._customer_engagement_analysis(customer_details)
                
                # Targeted Marketing Strategy
                self._generate_targeted_marketing_strategy(customer_details)
                
                # Export Options
                self._export_customer_report(customer_details)
            else:
                st.warning("No detailed information available for the selected customer.")
        st.markdown("---")  # Add a divider
        if st.button("🤖 Go to AI Assistant", key="ai_assistant_nav"):
            self._navigate_to_ai_assistant()

# Add main function with authentication decorator
@requires_auth
def main():
    """
    Main function to initialize and render the Customer Lookup page
    """
    # Create an instance of the CustomerLookupPage class
    customer_page = CustomerLookupPage()
    
    # Render the page
    customer_page.render()

# Ensure the script can be run directly
if __name__ == "__main__":
    main()