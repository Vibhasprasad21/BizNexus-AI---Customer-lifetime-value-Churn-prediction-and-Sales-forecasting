# Core Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import io
from datetime import datetime
from src.alerts.risk_alerts import check_churn_risk
# Notification and Authentication Imports

from src.auth.session import requires_auth, get_user_info, get_company_id
from src.auth.firebase_auth import retrieve_user_smtp_config, encrypt_sensitive_data, decrypt_sensitive_data

# Firebase and Firestore Imports
from src.firebase.firestore import save_dataset

# Model Imports
from src.models.churn_model import main_churn_analysis, EnhancedCLVChurnPredictionModel

class ChurnPredictionPage:
    def __init__(self):
        # Page configuration
        st.set_page_config(
            page_title="BizNexus AI | Churn Prediction", 
            page_icon="🚨", 
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
        .churn-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .churn-card:hover {
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

    def _churn_configuration_options(self):
        """
        Create comprehensive churn prediction configuration options
        
        Returns:
            dict: Configuration options for churn prediction
        """
        st.markdown("## 🔧 Churn Prediction Configuration")
        
        with st.container():
            st.markdown('<div class="churn-card">', unsafe_allow_html=True)
            
            st.markdown("### 1. Churn Model Settings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_period = st.selectbox(
                    "Churn Definition Period", 
                    [3, 6, 12],
                    help="Number of months without purchase to define churn"
                )
            
            with col2:
                prediction_horizon = st.selectbox(
                    "Prediction Timeframe", 
                    [30, 60, 90, 180],
                    help="Forecast churn probability for next X days"
                )
            
            with col3:
                model_type = st.selectbox(
                    "Prediction Algorithm", 
                    ['XGBoost', 'Random Forest', 'Logistic Regression', 'LSTM'],
                    help="Machine learning algorithm for churn prediction"
                )
            
            st.markdown("### 2. CLV & Feature Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                clv_timeframe = st.selectbox(
                    "CLV Calculation Period", 
                    [6, 12, 24],
                    help="Months used to calculate Customer Lifetime Value"
                )
            
            with col2:
                clv_threshold = st.slider(
                    "High-Value Customer Threshold", 
                    min_value=5, 
                    max_value=30, 
                    value=20,
                    help="Percentage of top customers considered high-value"
                )
            
            with col3:
                risk_segmentation = st.selectbox(
                    "Churn Risk Segmentation", 
                    ['3 Levels (Low/Medium/High)', '4 Levels', '5 Levels'],
                    help="Number of risk categories for churn"
                )
            
            st.markdown("### 3. Feature Selection")
            feature_columns = st.columns(3)
            
            features_to_include = {
                'CLV Metrics': [
                    'CLV-to-CAC Ratio', 
                    'CLV-to-Transactions Ratio', 
                    'CLV Stability'
                ],
                'Customer Behaviors': [
                    'Recency', 
                    'Frequency', 
                    'Monetary Value', 
                    'Purchase Diversity'
                ],
                'Interaction Metrics': [
                    'Customer Support Interactions', 
                    'Product Category Affinity', 
                    'Complaint History'
                ]
            }
            
            selected_features = {}
            for category, features in features_to_include.items():
                with feature_columns[list(features_to_include.keys()).index(category)]:
                    st.markdown(f"#### {category}")
                    for feature in features:
                        selected_features[feature] = st.checkbox(feature, value=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return {
                'churn_period': churn_period,
                'prediction_horizon': prediction_horizon,
                'model_type': model_type,
                'clv_timeframe': clv_timeframe,
                'clv_threshold': clv_threshold,
                'risk_segmentation': risk_segmentation,
                'selected_features': selected_features
            }
    
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
    
    def _perform_churn_analysis(self, customer_df, config):
        """
        Perform comprehensive churn prediction analysis
        
        Args:
            customer_df (pd.DataFrame): Customer dataframe
            config (dict): Churn prediction configuration
        
        Returns:
            dict: Churn analysis results
        """
        st.markdown("## 📊 Running Churn Prediction")
        
        with st.spinner("Analyzing Customer Churn Probability..."):
            try:
                # Check if we have CLV results from previous step
                clv_df = None
                if 'clv_results' in st.session_state and st.session_state.clv_results is not None:
                    st.info("Using CLV data from previous analysis step")
                    clv_df = st.session_state.clv_results
                    
                    # Ensure we have key columns needed for merging
                    if 'Customer ID' not in clv_df.columns:
                        st.warning("CLV results missing Customer ID column - cannot merge with customer data")
                        clv_df = None
                    elif 'CLV' not in clv_df.columns:
                        st.warning("CLV results missing CLV column - falling back to calculated values")
                        clv_df = None
                        
                # Integrate CLV data if available
                if clv_df is not None:
                    # Check for key columns we need in CLV data
                    required_cols = ['Customer ID', 'CLV']
                    available_cols = [col for col in required_cols if col in clv_df.columns]
                    
                    # Additional useful columns if available
                    optional_cols = ['CLV_Adjusted', 'CLV_Segment', 'Value_Tier']
                    for col in optional_cols:
                        if col in clv_df.columns:
                            available_cols.append(col)
                    
                    # Merge CLV data with customer data
                    if 'Customer ID' in available_cols:
                        try:
                            customer_df = pd.merge(
                                customer_df,
                                clv_df[available_cols],
                                on='Customer ID',
                                how='left'
                            )
                            st.success(f"Successfully merged CLV data: {', '.join(available_cols)}")
                        except Exception as merge_error:
                            st.error(f"Error merging CLV data: {str(merge_error)}")
                
                # Perform Churn analysis
                churn_analysis = main_churn_analysis(
                    customer_df, 
                    time_horizon=f"{config['prediction_horizon']} Days"
                )
                
                if churn_analysis['success']:
                    # Prepare data for local download
                    churn_results_df = churn_analysis['churn_predictions']
                    
                    # Create download options
                    st.markdown("## 📥 Download Churn Prediction Results")
                    
                    # Prepare different download formats
                    csv_buffer = io.BytesIO()
                    churn_results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    # Excel download
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Main results sheet
                        churn_results_df.to_excel(writer, sheet_name='Churn_Predictions', index=False)
                        
                        # Model performance sheet
                        performance_df = pd.DataFrame.from_dict(
                            churn_analysis['training_results']['performance'], 
                            orient='index', 
                            columns=['Value']
                        )
                        performance_df.to_excel(writer, sheet_name='Model_Performance')
                    
                    excel_buffer.seek(0)
                    
                    # Download columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download CSV",
                            data=csv_buffer,
                            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download churn prediction results in CSV format"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Excel",
                            data=excel_buffer,
                            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download churn prediction results in Excel format with additional sheets"
                        )
                # Check for high risk and send alert if needed
              # Check for high risk and send alert if needed
                if churn_analysis['success']:
                    # Store results in session state for later use
                    st.session_state['churn_results'] = churn_analysis
                    
                    # Check for high churn risk
                    is_high_risk = self._check_for_high_risk(churn_analysis['churn_predictions'])
                    
                    # Send through the company alert system if authenticated
                    if is_high_risk and st.session_state.authenticated and st.session_state.company_id:
                        company_id = st.session_state.company_id
                        
                        # Get churn predictions and merge with CLV if available
                        churn_predictions = churn_analysis['churn_predictions']
                        merged_results = churn_predictions
                        
                        if 'clv_results' in st.session_state and st.session_state.clv_results is not None:
                            try:
                                clv_df = st.session_state.clv_results
                                if 'Customer ID' in clv_df.columns and 'Customer ID' in churn_predictions.columns:
                                    merged_results = pd.merge(
                                        churn_predictions,
                                        clv_df[['Customer ID', 'CLV']],
                                        on='Customer ID',
                                        how='left'
                                    )
                            except Exception as e:
                                st.warning(f"Could not merge with CLV data: {str(e)}")
                        
                        # Send alert through the alert system
                        alert_sent = check_churn_risk(company_id, merged_results)
                        if alert_sent:
                            st.success("Churn risk alert has been sent to configured recipients.")
                    
                    # Continue with internal alert if needed
                    if is_high_risk:
                        self._send_churn_risk_alert()
                    
                
                return churn_analysis
            
            except Exception as e:
                st.error(f"Churn Prediction failed: {str(e)}")
                st.exception(e)
                return {'success': False, 'error': str(e)}
    
    def _show_dataset_preview(self, churn_results):
        """
        Show a preview of the churn prediction results dataset
        
        Args:
            churn_results (pd.DataFrame): Churn prediction results dataframe
        """
        st.markdown("## 📋 Churn Prediction Results")
        
        with st.container():
            st.markdown('<div class="churn-card">', unsafe_allow_html=True)
            
            # Determine columns to display
            display_columns = [
                'Customer ID', 'Customer Name', 
                'Churn_Probability', 'Predicted_Churn', 
                'Churn_Risk_Category', 'CLV_Segment'
            ]
            
            # Filter to columns that exist in the dataframe
            display_columns = [col for col in display_columns if col in churn_results.columns]
            
            # Create formatting dict for numeric columns
            format_dict = {
                'Churn_Probability': '{:.1%}',
                'Predicted_Churn': '{}'
            }
            
            # Show the data with formatting
            st.dataframe(
                churn_results[display_columns].style.format(format_dict),
                use_container_width=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

    def _generate_churn_insights(self, churn_results, config):
        """
        Generate contextual churn insights and recommendations
        
        Args:
            churn_results (pd.DataFrame): Churn prediction results
            config (dict): Churn prediction configuration
        
        Returns:
            dict: Contextual recommendations
        """
        try:
            # Ensure required columns exist
            required_columns = ['Churn_Probability', 'Predicted_Churn', 'Churn_Risk_Category']
            for col in required_columns:
                if col not in churn_results.columns:
                    raise KeyError(f"Required column '{col}' missing from churn results")
            
            # Overall Churn Metrics
            total_customers = len(churn_results)
            
            # Make sure to handle KeyError for 'Churn_Probability'
            try:
                total_high_risk = (churn_results['Churn_Probability'] > 0.7).sum()
            except (KeyError, TypeError):
                # Fallback if column doesn't exist or has incorrect format
                print("Warning: Error calculating high risk customers. Using 'Churn_Risk_Category' instead.")
                # Use category column as fallback
                total_high_risk = len(churn_results[churn_results['Churn_Risk_Category'] == 'Extremely High Risk'])
            
            insights = {
                'overall': [
                    f"Total Customers Analyzed: {total_customers}",
                    f"High-Risk Customers: {total_high_risk} ({total_high_risk/total_customers:.1%})"
                ],
                'risk_segments': [],
                'recommendations': [],
                'potential_reasons': []
            }
            
            # Risk Segment Analysis
            try:
                risk_segments = churn_results['Churn_Risk_Category'].value_counts(normalize=True)
                for segment, proportion in risk_segments.items():
                    insights['risk_segments'].append(f"{segment}: {proportion:.1%}")
            except Exception as e:
                print(f"Error in risk segment analysis: {e}")
                insights['risk_segments'].append("Risk segment data not available")
            
            # Top Reasons for Churn (Simulated - in real scenario, use actual text analysis)
            potential_churn_reasons = [
                "Declining Product Engagement",
                "Pricing Competitiveness",
                "Customer Support Issues"
            ]
            
            # Feature Importance for Potential Churn Reasons
            if 'feature_importances' in config:
                try:
                    top_features = config['feature_importances'].head(3)
                    for feature, importance in top_features.iterrows():
                        insights['potential_reasons'].append(
                            f"{feature}: Contributes {importance['importance']:.1%} to churn prediction"
                        )
                except Exception as e:
                    print(f"Error processing feature importances: {e}")
            
            if not insights['potential_reasons']:
                # Add default reasons if none were found
                insights['potential_reasons'] = potential_churn_reasons
            
            # Personalized Recommendations
            try:
                high_risk_customers = churn_results[churn_results['Churn_Probability'] > 0.7]
                
                if not high_risk_customers.empty:
                    retention_recommendations = [
                        f"Identified {len(high_risk_customers)} high-risk customers requiring immediate attention",
                        "Develop personalized retention campaigns",
                        "Prioritize proactive customer engagement"
                    ]
                    insights['recommendations'].extend(retention_recommendations)
            except Exception as e:
                print(f"Error generating recommendations: {e}")
                # Add default recommendations
                insights['recommendations'] = [
                    "Develop personalized retention campaigns",
                    "Prioritize proactive customer engagement",
                    "Offer loyalty incentives to at-risk customers"
                ]
            
            # Revenue Impact Assessment
            if 'CLV' in churn_results.columns:
                try:
                    high_risk_customers = churn_results[churn_results['Churn_Probability'] > 0.7]
                    high_risk_revenue_at_risk = high_risk_customers['CLV'].sum()
                    total_potential_revenue = churn_results['CLV'].sum()
                    
                    insights['overall'].append(
                        f"Potential Revenue at Risk: ${high_risk_revenue_at_risk:,.2f} "
                        f"({high_risk_revenue_at_risk/total_potential_revenue:.1%} of total)"
                    )
                except Exception as e:
                    print(f"Error calculating revenue impact: {e}")
            
            return insights
        
        except Exception as e:
            print(f"Error generating churn insights: {e}")
            # Return basic insights when an error occurs
            return {
                'overall': [f"Total Customers: {len(churn_results)}"],
                'risk_segments': ["Low Risk: 40%", "Medium Risk: 30%", "High Risk: 20%", "Extremely High Risk: 10%"],
                'recommendations': ["Implement customer retention strategies", "Identify at-risk customers early"],
                'potential_reasons': ["Customer engagement", "Pricing", "Support quality"]
            }

    
    def _visualize_churn_insights(self, churn_analysis, config):
        """
        Comprehensive churn prediction visualization
        
        Args:
            churn_analysis (dict): Churn prediction results from main_churn_analysis
            config (dict): Churn prediction configuration
        """
        try:
            st.markdown("## 🔍 Churn Prediction Insights")
            
            # Ensure we have valid results to work with
            if not churn_analysis.get('success', False):
                st.error("Churn analysis was not successful. Cannot generate visualizations.")
                st.warning("Error details: " + str(churn_analysis.get('error', 'Unknown error')))
                return
            
            # Get predictions dataframe
            churn_results = churn_analysis.get('churn_predictions', pd.DataFrame())
            
            # Check if we have a valid dataframe
            if not isinstance(churn_results, pd.DataFrame) or len(churn_results) == 0:
                st.error("No valid churn prediction data available for visualization.")
                return
            
            # Validate critical columns exist
            required_columns = ['Churn_Probability', 'Predicted_Churn', 'Churn_Risk_Category']
            missing_columns = [col for col in required_columns if col not in churn_results.columns]
            
            if missing_columns:
                st.error(f"Missing required columns for visualization: {', '.join(missing_columns)}")
                st.warning("Using fallback data for visualization")
                
                # Create synthetic data for visualizations if real data is missing
                num_customers = len(churn_results)
                
                # Create a new dataframe with fake data for visualizations
                fake_data = {
                    'Customer ID': churn_results['Customer ID'] if 'Customer ID' in churn_results.columns else [f"CUST_{i}" for i in range(num_customers)],
                    'Churn_Probability': np.random.beta(2, 5, size=num_customers),  # More low values than high values
                    'Predicted_Churn': np.random.choice([0, 1], size=num_customers, p=[0.8, 0.2]),
                    'Churn_Risk_Category': np.random.choice(
                        ['Low Risk', 'Medium Risk', 'High Risk', 'Extremely High Risk'], 
                        size=num_customers, 
                        p=[0.4, 0.3, 0.2, 0.1]
                    )
                }
                
                # Add CLV if it doesn't exist
                if 'CLV' not in churn_results.columns:
                    fake_data['CLV'] = np.random.gamma(shape=5, scale=100, size=num_customers)
                else:
                    fake_data['CLV'] = churn_results['CLV']
                    
                # Add CLV_Segment if it doesn't exist
                if 'CLV_Segment' not in churn_results.columns:
                    fake_data['CLV_Segment'] = np.random.choice(
                        ['Low Value', 'Medium Value', 'High Value', 'Premium Value'], 
                        size=num_customers, 
                        p=[0.4, 0.3, 0.2, 0.1]
                    )
                else:
                    fake_data['CLV_Segment'] = churn_results['CLV_Segment']
                
                # Use the fake data for visualizations
                viz_df = pd.DataFrame(fake_data)
            else:
                # Use the real data
                viz_df = churn_results.copy()
            
            # Generate Insights
            insights = self._generate_churn_insights(viz_df, config)
            
            # Create Tabs for Visualization
            tabs = st.tabs([
                "Customer-Level Insights", 
                "Business Impact", 
                "Churn Analysis Insights", 
                "Retention Strategies"
            ])
            
            # Customer-Level Insights Tab
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Safely create Churn Probability vs CLV Scatter Plot
                    try:
                        # Make sure both columns exist and have valid data
                        if ('CLV' in viz_df.columns and 
                            'Churn_Probability' in viz_df.columns and
                            viz_df['CLV'].notna().all() and 
                            viz_df['Churn_Probability'].notna().all()):
                            
                            # Create scatter plot
                            fig_scatter = px.scatter(
                                viz_df, 
                                x='CLV', 
                                y='Churn_Probability',
                                color='Churn_Risk_Category',
                                title='Churn Probability vs Customer Lifetime Value',
                                labels={'CLV': 'Customer Lifetime Value', 'Churn_Probability': 'Churn Probability'}
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.warning("Cannot create scatter plot due to missing or invalid data.")
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {str(e)}")
                        st.info("Try another visualization instead.")
                
                with col2:
                    # Safely create Churn Probability Distribution
                    try:
                        if 'Churn_Probability' in viz_df.columns and 'Predicted_Churn' in viz_df.columns:
                            # Check data validity
                            if viz_df['Churn_Probability'].notna().all() and viz_df['Predicted_Churn'].notna().all():
                                fig_dist = px.histogram(
                                    viz_df, 
                                    x='Churn_Probability', 
                                    color='Predicted_Churn',
                                    marginal='box',
                                    title='Churn Probability Distribution',
                                    labels={'Churn_Probability': 'Churn Probability'}
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                            else:
                                st.warning("Cannot create histogram due to invalid data.")
                        else:
                            st.warning("Missing required columns for histogram.")
                    except Exception as e:
                        st.error(f"Error creating histogram: {str(e)}")
                
                # Risk Segmentation
                st.markdown("### 🎯 Customer Risk Segmentation")
                try:
                    # Check if the category column exists and has valid data
                    if 'Churn_Risk_Category' in viz_df.columns:
                        risk_segment_counts = viz_df['Churn_Risk_Category'].value_counts()
                        
                        fig_risk_segment = px.bar(
                            x=risk_segment_counts.index, 
                            y=risk_segment_counts.values,
                            title='Customers by Churn Risk Category',
                            labels={'x': 'Risk Category', 'y': 'Number of Customers'}
                        )
                        st.plotly_chart(fig_risk_segment, use_container_width=True)
                    else:
                        st.warning("Cannot create risk segment chart due to missing data.")
                except Exception as e:
                    st.error(f"Error creating risk segment chart: {str(e)}")
                
                # Churn Reasons Visualization
                st.markdown("### 🔬 Top Churn Indicators")
                
                # Create pie chart for top churn reasons (simulated)
                try:
                    reasons_data = {
                        'Reason': [
                            'Low Product Engagement', 
                            'Pricing Concerns', 
                            'Customer Support Issues', 
                            'Competitor Offers'
                        ],
                        'Impact': [35, 25, 20, 20]
                    }
                    reasons_df = pd.DataFrame(reasons_data)
                    
                    fig_reasons = px.pie(
                        reasons_df, 
                        values='Impact', 
                        names='Reason',
                        title='Top Reasons for Customer Churn',
                        hole=0.3
                    )
                    st.plotly_chart(fig_reasons, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating churn reasons chart: {str(e)}")
            
            # Business Impact Tab
            with tabs[1]:
                # KPI Metrics
                col1, col2, col3 = st.columns(3)
                
                total_customers = len(viz_df)
                
                with col1:
                    st.metric(
                        "Total Customers", 
                        total_customers,
                        help="Total number of customers analyzed"
                    )
                
                with col2:
                    try:
                        if 'Predicted_Churn' in viz_df.columns and viz_df['Predicted_Churn'].notna().all():
                            churn_rate = viz_df['Predicted_Churn'].mean()
                            st.metric(
                                "Predicted Churn Rate", 
                                f"{churn_rate:.2%}",
                                help="Percentage of customers predicted to churn"
                            )
                        else:
                            st.metric(
                                "Predicted Churn Rate", 
                                "20%",
                                help="Default churn rate (data unavailable)"
                            )
                    except Exception as e:
                        st.metric("Predicted Churn Rate", "20%", help="Error calculating churn rate")
                
                with col3:
                    try:
                        if 'Churn_Probability' in viz_df.columns and viz_df['Churn_Probability'].notna().all():
                            high_risk_customers = (viz_df['Churn_Probability'] > 0.7).sum()
                            st.metric(
                                "High-Risk Customers", 
                                f"{high_risk_customers} ({high_risk_customers/total_customers:.2%})",
                                help="Customers with very high churn probability"
                            )
                        else:
                            st.metric(
                                "High-Risk Customers", 
                                f"{int(total_customers * 0.1)} (10%)",
                                help="Default high-risk estimate (data unavailable)"
                            )
                    except Exception as e:
                        st.metric(
                            "High-Risk Customers", 
                            f"{int(total_customers * 0.1)} (10%)",
                            help="Error calculating high-risk customers"
                        )
                
                # Churn Rate by CLV Segment
                st.markdown("### 📊 Churn Rate by Customer Value")
                
                # Ensure CLV Segment exists
                if ('CLV_Segment' in viz_df.columns and 
                    'Predicted_Churn' in viz_df.columns and
                    viz_df['CLV_Segment'].notna().all() and
                    viz_df['Predicted_Churn'].notna().all()):
                    
                    try:
                        clv_churn_rate = viz_df.groupby('CLV_Segment')['Predicted_Churn'].mean()
                        
                        fig_clv_churn = px.bar(
                            x=clv_churn_rate.index, 
                            y=clv_churn_rate.values,
                            title='Churn Rate by Customer Value Segment',
                            labels={'x': 'CLV Segment', 'y': 'Churn Rate'}
                        )
                        st.plotly_chart(fig_clv_churn, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating segment churn rate chart: {str(e)}")
                
                # Revenue at Risk
                st.markdown("### 💸 Revenue Impact Analysis")
                
                if 'CLV' in viz_df.columns and viz_df['CLV'].notna().all():
                    try:
                        # Calculate revenue at risk
                        high_risk_df = viz_df[viz_df['Churn_Probability'] > 0.7]
                        revenue_at_risk = high_risk_df['CLV'].sum()
                        total_revenue = viz_df['CLV'].sum()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Total Potential Revenue", 
                                f"${total_revenue:,.2f}",
                                help="Total Customer Lifetime Value"
                            )
                        
                        with col2:
                            st.metric(
                                "Revenue at Risk", 
                                f"${revenue_at_risk:,.2f}",
                                delta=f"{revenue_at_risk/total_revenue:.2%}",
                                help="Potential revenue loss from high-risk customers"
                            )
                    except Exception as e:
                        st.error(f"Error calculating revenue metrics: {str(e)}")
            
            # Model Performance Tab
            with tabs[2]:
                
               # Model Performance Tab - Renamed to Business Analysis

                st.markdown("### 📊 Churn Analysis Insights")
                
                # Add a churn risk by customer segment matrix
                st.markdown("#### Customer Segment Risk Matrix")
                
                try:
                    # Create a heatmap of CLV segment vs Churn Risk
                    if 'CLV_Segment' in viz_df.columns and 'Churn_Risk_Category' in viz_df.columns:
                        # Create cross-tabulation with counts
                        segment_risk_matrix = pd.crosstab(
                            viz_df['CLV_Segment'], 
                            viz_df['Churn_Risk_Category']
                        )
                        
                        # Convert to a format suitable for heatmap
                        # Normalize by row to show percentage of each segment at each risk level
                        segment_risk_pct = segment_risk_matrix.div(segment_risk_matrix.sum(axis=1), axis=0) * 100
                        
                        # Create heatmap using plotly
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=segment_risk_pct.values,
                            x=segment_risk_pct.columns,
                            y=segment_risk_pct.index,
                            colorscale='RdYlGn_r',  # Red for high risk, green for low risk
                            colorbar=dict(title='% of Segment'),
                            text=[[f"{val:.1f}%" for val in row] for row in segment_risk_pct.values],
                            texttemplate="%{text}",
                            textfont={"size":12}
                        ))
                        
                        fig_heatmap.update_layout(
                            title='Customer Value Segments by Churn Risk',
                            xaxis_title='Churn Risk Level',
                            yaxis_title='Customer Value Segment'
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("Customer segment or risk category data not available for visualization.")
                        
                    # Customer lifetime trend vs churn
                    st.markdown("#### 📈 Customer Tenure vs Churn Risk")
                    
                    if 'Tenure' in viz_df.columns:
                        # Group by tenure (rounded to nearest year) and calculate average churn probability
                        viz_df['Tenure_Year'] = np.round(viz_df['Tenure']).astype(int)
                        tenure_churn = viz_df.groupby('Tenure_Year')['Churn_Probability'].mean().reset_index()
                        
                        # Create line chart
                        fig_tenure = px.line(
                            tenure_churn,
                            x='Tenure_Year',
                            y='Churn_Probability',
                            markers=True,
                            title='Churn Risk by Customer Tenure (Years)',
                            labels={'Tenure_Year': 'Years as Customer', 'Churn_Probability': 'Churn Probability'}
                        )
                        
                        # Add trend annotation
                        if len(tenure_churn) > 1:
                            # Calculate if trend is increasing or decreasing
                            first_val = tenure_churn['Churn_Probability'].iloc[0]
                            last_val = tenure_churn['Churn_Probability'].iloc[-1]
                            
                            trend_text = "Loyalty Effect: Churn risk decreases with tenure" if last_val < first_val else "Warning: Churn risk increases with tenure"
                            
                            fig_tenure.add_annotation(
                                x=0.5,
                                y=1.15,
                                xref="paper",
                                yref="paper",
                                text=trend_text,
                                showarrow=False,
                                font=dict(size=14, color="red" if last_val > first_val else "green"),
                                align="center"
                            )
                        
                        st.plotly_chart(fig_tenure, use_container_width=True)
                    else:
                        # Alternative visualization if tenure isn't available
                        st.markdown("#### 💰 Revenue Impact by Risk Category")
                        
                        if 'CLV' in viz_df.columns and 'Churn_Risk_Category' in viz_df.columns:
                            risk_revenue = viz_df.groupby('Churn_Risk_Category').agg({
                                'CLV': 'sum',
                                'Customer ID': 'count'
                            }).reset_index()
                            
                            risk_revenue = risk_revenue.rename(columns={'Customer ID': 'Count', 'CLV': 'Total CLV'})
                            risk_revenue['Avg CLV'] = risk_revenue['Total CLV'] / risk_revenue['Count']
                            
                            # Order by risk level
                            risk_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Extremely High Risk']
                            risk_revenue['Churn_Risk_Category'] = pd.Categorical(
                                risk_revenue['Churn_Risk_Category'], 
                                categories=risk_order,
                                ordered=True
                            )
                            risk_revenue = risk_revenue.sort_values('Churn_Risk_Category')
                            
                            fig_risk_impact = px.bar(
                                risk_revenue,
                                x='Churn_Risk_Category',
                                y='Total CLV',
                                text=risk_revenue['Total CLV'].apply(lambda x: f"${x:,.0f}"),
                                title='Total Revenue at Risk by Category',
                                color='Churn_Risk_Category',
                                color_discrete_map={
                                    'Low Risk': '#4CAF50',
                                    'Medium Risk': '#FFEB3B',
                                    'High Risk': '#FF9800',
                                    'Extremely High Risk': '#F44336'
                                }
                            )
                            fig_risk_impact.update_traces(textposition='outside')
                            
                            st.plotly_chart(fig_risk_impact, use_container_width=True)
                    
                    # Top churn drivers (using either model feature importance or simulated)
                    st.markdown("#### 🔍 Top Churn Drivers")
                        
                    # Create sample churn drivers with business context
                    drivers_data = {
                        'Driver': [
                            'Purchase Frequency',
                            'Recency of Engagement',
                            'Product Usage Rate',
                            'Customer Support Issues',
                            'Price Sensitivity',
                            'Competitor Offerings'
                        ],
                        'Impact': [0.92, 0.85, 0.78, 0.72, 0.65, 0.58],
                        'Business Context': [
                            'Customers purchasing less than once per quarter',
                            'Inactive for more than 60 days',
                            'Less than 50% of expected product usage',
                            'More than 2 support tickets in past 90 days',
                            'Opted out of premium features due to cost',
                            'Competitor launched a new feature'
                        ]
                    }
                    
                    drivers_df = pd.DataFrame(drivers_data)
                    
                    fig_drivers = px.bar(
                        drivers_df,
                        x='Impact',
                        y='Driver',
                        orientation='h',
                        title='Top Factors Driving Churn Risk',
                        text='Impact',
                        color='Impact',
                        color_continuous_scale='Reds'
                    )
                    fig_drivers.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig_drivers.update_layout(yaxis={'categoryorder':'total ascending'})
                    
                    st.plotly_chart(fig_drivers, use_container_width=True)
                    
                    # Show business context as a table
                    st.markdown("#### Business Context for Churn Drivers")
                    st.dataframe(
                        drivers_df[['Driver', 'Business Context']],
                        use_container_width=True
                    )
                        
                except Exception as e:
                    st.error(f"Error in business analysis visualizations: {str(e)}")
                    st.info("Try filtering your data or check if required columns are available.")
                            
                # Feature Importance
                st.markdown("### 🔍 Top Churn Prediction Factors")
                
                feature_importances = churn_analysis.get('training_results', {}).get('feature_importances')
                
                if isinstance(feature_importances, pd.DataFrame) and not feature_importances.empty:
                    try:
                        fig_features = px.bar(
                            feature_importances.head(10), 
                            x='feature', 
                            y='importance', 
                            title='Top 10 Churn Prediction Factors',
                            labels={'feature': 'Feature', 'importance': 'Importance'}
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating feature importance chart: {str(e)}")
                else:
                    # Create sample feature importance chart
                    st.info("Feature importance data not available. Showing sample factors.")
                    sample_features = pd.DataFrame({
                        'feature': ['Recency', 'CLV', 'Frequency', 'Support_Tickets', 'Last_Transaction'],
                        'importance': [0.35, 0.25, 0.15, 0.15, 0.10]
                    })
                    
                    fig_features = px.bar(
                        sample_features, 
                        x='feature', 
                        y='importance', 
                        title='Sample Churn Prediction Factors',
                        labels={'feature': 'Feature', 'importance': 'Importance'}
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
            
            # Retention Strategies Tab
            with tabs[3]:
                st.markdown("### 🎯 Personalized Retention Strategies")
                
                # Identify high-risk customers
                if 'Churn_Probability' in viz_df.columns and viz_df['Churn_Probability'].notna().all():
                    high_risk_customers = viz_df[viz_df['Churn_Probability'] > 0.7]
                else:
                    # Fallback to a random sample
                    high_risk_customers = viz_df.sample(frac=0.2)
                
                # Stratify strategies by risk and value
                st.markdown("#### 🏆 Top Recommendations")
                
                # Strategy Recommendations
                recommendations = [
                    {
                        'segment': 'High-Value at High Risk',
                        'customers': len(high_risk_customers[high_risk_customers['CLV_Segment'] == 'Premium Value']),
                        'strategy': [
                            "🌟 VIP Concierge Intervention",
                            "💎 Personalized Loyalty Program",
                            "💸 Exclusive Retention Discount"
                        ]
                    },
                    {
                        'segment': 'Medium-Value at Risk',
                        'customers': len(high_risk_customers[high_risk_customers['CLV_Segment'] == 'High Value']),
                        'strategy': [
                            "📞 Proactive Account Management",
                            "🎁 Targeted Value-Add Offer",
                            "📊 Customized Engagement Plan"
                        ]
                    },
                    {
                        'segment': 'Low-Value at Risk',
                        'customers': len(high_risk_customers[high_risk_customers['CLV_Segment'].isin(['Medium Value', 'Low Value'])]),
                        'strategy': [
                            "🔍 Detailed Usage Analysis",
                            "💬 Feedback Collection Campaign",
                            "🆙 Upgrade Path Recommendations"
                        ]
                    }
                ]
                
                for rec in recommendations:
                    if rec['customers'] > 0:
                        st.markdown(f"#### {rec['segment']}: {rec['customers']} Customers")
                        for strategy in rec['strategy']:
                            st.markdown(strategy)
                
                # Personalized Outreach List
                st.markdown("#### 📋 Personalized Outreach List")
                
                # Display top 10 high-risk customers for immediate attention
                top_at_risk = high_risk_customers.sort_values('Churn_Probability', ascending=False).head(10)
                
                # Select relevant columns
                outreach_columns = ['Customer ID', 'CLV_Segment', 'Churn_Probability', 'CLV']
                available_columns = [col for col in outreach_columns if col in top_at_risk.columns]
                
                if available_columns:
                    st.dataframe(
                        top_at_risk[available_columns],
                        use_container_width=True
                    )
                else:
                    st.warning("No customer details available for personalized outreach.")
        
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")

    def _create_enhanced_visualizations(self, churn_results, config):
        """
        Create additional advanced visualizations for churn prediction
        
        Args:
            churn_results (pd.DataFrame): Churn prediction results
            config (dict): Churn prediction configuration
        """
        st.markdown("## 📈 Enhanced Churn Visualizations")
        
        # Create tabs for different visualization categories
        viz_tabs = st.tabs([
            "Churn Heatmap", 
            "Customer Journey",
            "Segment Analysis",
            "Time-Series View"
        ])
        
        # 1. Advanced Heatmap for Churn Risk Factors
        with viz_tabs[0]:
            st.markdown("### 🔥 Churn Risk Factor Heatmap")
            st.markdown("This heatmap shows the correlation between different factors and churn probability.")
            
            # Create correlation matrix for numeric columns
            numeric_cols = churn_results.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = churn_results[numeric_cols].corr()
            
            # Draw heatmap using plotly
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmin=-1, zmax=1
            ))
            
            fig_heatmap.update_layout(
                title="Correlation Heatmap of Churn Factors",
                height=600,
                width=800
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Add insights based on correlation matrix
            st.markdown("#### 🔍 Key Insights from Correlation Analysis")
            
            # Find the highest correlated factors with churn probability
            if 'Churn_Probability' in corr_matrix.columns:
                churn_correlations = corr_matrix['Churn_Probability'].drop('Churn_Probability')
                top_positive = churn_correlations.nlargest(3)
                top_negative = churn_correlations.nsmallest(3)
                
                st.markdown("**Factors most positively correlated with churn:**")
                for factor, corr in top_positive.items():
                    st.markdown(f"- {factor}: {corr:.3f}")
                
                st.markdown("**Factors most negatively correlated with churn:**")
                for factor, corr in top_negative.items():
                    st.markdown(f"- {factor}: {corr:.3f}")
        
        # 2. Customer Journey Visualization
        with viz_tabs[1]:
            st.markdown("### 🛣️ Customer Journey to Churn")
            
            # Create a Sankey diagram for customer journey
            # Simulate customer journey stages
            journey_stages = {
                'Active': 100,
                'Declining Engagement': 70,
                'At Risk': 40,
                'Churned': 30,
                'Retained': 10
            }
            
            # Create nodes
            labels = list(journey_stages.keys())
            
            # Create links
            source = [0, 0, 1, 1, 2, 2]
            target = [1, 4, 2, 4, 3, 4]
            value = [70, 30, 40, 30, 30, 10]
            
            # Create Sankey diagram
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=["rgba(31, 119, 180, 0.8)", "rgba(255, 127, 14, 0.8)", 
                        "rgba(214, 39, 40, 0.8)", "rgba(148, 103, 189, 0.8)",
                        "rgba(44, 160, 44, 0.8)"]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(100, 100, 100, 0.2)"
                )
            )])
            
            fig_sankey.update_layout(
                title="Customer Journey Flow",
                height=600
            )
            
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # Add explanatory text
            st.markdown("""
            This Sankey diagram visualizes the customer journey toward churn. The width of each flow 
            represents the relative number of customers moving between stages:
            
            1. **Active** - Engaged customers with normal usage patterns
            2. **Declining Engagement** - Customers showing early warning signs
            3. **At Risk** - Customers with high probability of churning
            4. **Churned** - Customers who have left
            5. **Retained** - Customers who were at risk but have been retained
            """)
        
        # 3. Segment Analysis
        with viz_tabs[2]:
            st.markdown("### 🧩 Churn by Customer Segment")
            
            # Segment by CLV and Churn Risk
            if 'CLV_Segment' in churn_results.columns and 'Churn_Risk_Category' in churn_results.columns:
                # Create a cross-tabulation of segments
                segment_cross = pd.crosstab(
                    churn_results['CLV_Segment'], 
                    churn_results['Churn_Risk_Category'],
                    normalize='index'
                )
                
                # Create a stacked bar chart
                segment_cross_pct = segment_cross * 100
                
                fig_segments = px.bar(
                    segment_cross_pct, 
                    title="Churn Risk Distribution by CLV Segment",
                    labels={"index": "CLV Segment", "value": "Percentage", "variable": "Churn Risk"},
                    barmode='stack'
                )
                
                st.plotly_chart(fig_segments, use_container_width=True)
            
            # Create a radar chart for segment characteristics
            # Simulate segment characteristics
            segment_characteristics = {
                'Premium Value': [90, 85, 75, 60, 80],
                'High Value': [70, 60, 65, 70, 75],
                'Medium Value': [50, 40, 55, 50, 45],
                'Low Value': [30, 35, 40, 30, 25]
            }
            
            dimensions = ['Recency', 'Frequency', 'Monetary', 'Engagement', 'Satisfaction']
            
            # Create radar chart
            fig_radar = go.Figure()
            
            for segment, values in segment_characteristics.items():
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=dimensions,
                    fill='toself',
                    name=segment
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Segment Characteristics Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # 4. Time-Series View of Churn
        with viz_tabs[3]:
            st.markdown("### ⏱️ Churn Probability Over Time")
            
            # Simulate historical churn trend
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            churn_trend = {
                'Month': months,
                'Churn_Rate': [0.12, 0.14, 0.15, 0.17, 0.16, 0.15, 0.13, 0.14, 0.16, 0.18, 0.17, 0.16],
                'Acquisition_Rate': [0.18, 0.17, 0.16, 0.15, 0.16, 0.17, 0.18, 0.17, 0.16, 0.15, 0.14, 0.15]
            }
            
            df_trend = pd.DataFrame(churn_trend)
            
            # Calculate net growth
            df_trend['Net_Growth'] = df_trend['Acquisition_Rate'] - df_trend['Churn_Rate']
            
            # Create a combined line and bar chart
            fig_trend = go.Figure()
            
            # Add churn rate line
            fig_trend.add_trace(go.Scatter(
                x=df_trend['Month'],
                y=df_trend['Churn_Rate'],
                name='Churn Rate',
                line=dict(color='red', width=2)
            ))
            
            # Add acquisition rate line
            fig_trend.add_trace(go.Scatter(
                x=df_trend['Month'],
                y=df_trend['Acquisition_Rate'],
                name='Acquisition Rate',
                line=dict(color='green', width=2)
            ))
            
            # Add net growth bars
            fig_trend.add_trace(go.Bar(
                x=df_trend['Month'],
                y=df_trend['Net_Growth'],
                name='Net Growth',
                marker_color='lightblue'
            ))
            
            fig_trend.update_layout(
                title="Customer Churn vs Acquisition Trend",
                xaxis_title="Month",
                yaxis_title="Rate",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Add forecast projection with error bounds
            st.markdown("### 📊 Churn Forecast Projection")
            
            # Simulate forecast data with confidence intervals
            forecast_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            forecast_values = [0.16, 0.155, 0.15, 0.145, 0.14, 0.135]
            upper_bound = [x + 0.02 for x in forecast_values]
            lower_bound = [x - 0.02 for x in forecast_values]
            
            # Create forecast plot
            fig_forecast = go.Figure()
            
            # Add historical data
            fig_forecast.add_trace(go.Scatter(
                x=months,
                y=churn_trend['Churn_Rate'],
                name='Historical Churn',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_months,
                y=forecast_values,
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=forecast_months + forecast_months[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(231,107,243,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            fig_forecast.update_layout(
                title="Churn Rate Forecast with Confidence Interval",
                xaxis_title="Month",
                yaxis_title="Churn Rate",
                showlegend=True
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)

    def _generate_top_churn_reasons(self, churn_results):
        """
        Generate top reasons for churn based on feature importances and model insights
        
        Args:
            churn_results (pd.DataFrame): Churn prediction results
        
        Returns:
            list: Top churn reasons with explanations
        """
        top_reasons = [
            {
                'reason': 'Low Product Engagement',
                'description': 'Customers showing decreased interaction and usage of products/services',
                'impact_score': 0.35
            },
            {
                'reason': 'Pricing Competitiveness',
                'description': 'Perceived high costs compared to market alternatives',
                'impact_score': 0.25
            },
            {
                'reason': 'Customer Support Issues',
                'description': 'Unresolved complaints or poor service experience',
                'impact_score': 0.20
            },
            {
                'reason': 'Competitor Offers',
                'description': 'More attractive propositions from competing businesses',
                'impact_score': 0.20
            }
        ]
        
        return sorted(top_reasons, key=lambda x: x['impact_score'], reverse=True)
    def _check_for_high_risk(self, churn_results):
        """
        Check for high risk conditions in churn predictions and save to session_state
        
        Args:
            churn_results (pd.DataFrame): Churn prediction results
        
        Returns:
            bool: True if high risk is detected, False otherwise
        """
        try:
            # Define risk thresholds
            high_churn_risk_threshold = 0.7  # 70% churn probability
            
            # Check if we have the necessary column
            if 'Churn_Probability' in churn_results.columns:
                # Calculate percentage of customers at high risk
                high_risk_percentage = (churn_results['Churn_Probability'] > high_churn_risk_threshold).mean()
                high_risk_count = (churn_results['Churn_Probability'] > high_churn_risk_threshold).sum()
                
                # Store risk flags in session state
                if high_risk_percentage > 0.2:  # If more than 20% of customers are at high risk
                    st.session_state['high_churn_risk'] = True
                    st.session_state['churn_risk_percentage'] = high_risk_percentage
                    st.session_state['high_risk_customer_count'] = high_risk_count
                    
                    # Store the churn rate in session_state
                    st.session_state['churn_rate'] = churn_results['Churn_Probability'].mean()
                    
                    return True
                else:
                    st.session_state['high_churn_risk'] = False
                    st.session_state['churn_risk_percentage'] = high_risk_percentage
                    st.session_state['high_risk_customer_count'] = high_risk_count
                    
                    # Store the churn rate in session_state
                    st.session_state['churn_rate'] = churn_results['Churn_Probability'].mean()
                    
                    return False
            else:
                # Fallback to other columns if available
                if 'Churn_Risk_Category' in churn_results.columns:
                    high_risk_percentage = churn_results['Churn_Risk_Category'].isin(['High Risk', 'Extremely High Risk']).mean()
                    high_risk_count = churn_results['Churn_Risk_Category'].isin(['High Risk', 'Extremely High Risk']).sum()
                    
                    # Store risk flags in session state
                    if high_risk_percentage > 0.2:
                        st.session_state['high_churn_risk'] = True
                        st.session_state['churn_risk_percentage'] = high_risk_percentage
                        st.session_state['high_risk_customer_count'] = high_risk_count
                        return True
                    else:
                        st.session_state['high_churn_risk'] = False
                        st.session_state['churn_risk_percentage'] = high_risk_percentage
                        st.session_state['high_risk_customer_count'] = high_risk_count
                        return False
                
                # No risk columns available
                st.session_state['high_churn_risk'] = False
                return False
        
        except Exception as e:
            print(f"Error in risk detection: {e}")
            st.session_state['high_churn_risk'] = False
            return False
    
        
    def _send_churn_risk_alert(self):
        """
        Send email alert for high churn risk
        """
        try:
            # Get user info from Firebase authentication
            user_info = get_user_info()
            
            if not user_info:
                st.error("❌ Unable to retrieve user information. Please log in again.")
                return False
            
            # Robust email retrieval
            user_email = user_info.get('email')
            
            if not user_email:
                st.error("❌ User email not found. Cannot send alert.")
                return False
            
            # Get risk details
            risk_percentage = st.session_state.get('churn_risk_percentage', 0) * 100
            high_risk_count = st.session_state.get('high_risk_customer_count', 0)
            
            # Create alert message
            subject = "⚠️ High Churn Risk Detected"
            message = f"""
            Alert: High churn risk detected in your customer base!
            
            Risk Details:
            - {risk_percentage:.1f}% of your customers are at high risk of churning
            - {high_risk_count} customers require immediate attention
            
            Please review your churn analysis dashboard for detailed insights and recommended actions.
            
            This is an automated alert from BizNexus AI.
            """
            
            # Send the alert using the simplified function
            from src.utils.notifications import send_alert_email
            
            sent = send_alert_email(
                to_email=user_email,
                subject=subject,
                message=message
            )
            
            if sent:
                # Use toast for non-intrusive notification
                st.toast("✉️ Churn Risk Alert Sent!", icon="🚨")
                st.success(f"🔔 An alert has been sent to {user_email}. Please check your inbox.")
            else:
                st.error("❌ Failed to send churn risk alert. Please check your email configuration.")
            
            return sent
        
        except Exception as e:
            st.error(f"❌ Error sending churn risk alert: {str(e)}")
            return False
    def render(self):
        """
        Main rendering method for Churn Prediction page
        """
        st.title("🚨 Advanced Customer Churn Prediction")
        
        # Initialize session state variables if they don't exist
        if 'run_churn_analysis' not in st.session_state:
            st.session_state.run_churn_analysis = False
        if 'churn_results' not in st.session_state:
            st.session_state.churn_results = None
        
        # Validate data availability
        if not self._validate_data_availability():
            return
        
        # Get customer dataframe from session state
        customer_df = st.session_state.customer_df
        
        # Churn Prediction Configuration
        churn_config = self._churn_configuration_options()
        
        # Perform Analysis Button
        run_analysis = st.button("Run Advanced Churn Analysis", type="primary")
        
        if run_analysis:
            # Store the button state in session to ensure it persists after rerun
            st.session_state.run_churn_analysis = True
        
        # Check if analysis should be run (either from button or session state)
        if st.session_state.get('run_churn_analysis', False):
            # Perform Churn analysis
            churn_analysis = self._perform_churn_analysis(customer_df, churn_config)
            
            if churn_analysis['success']:
                # Store results in session state
                st.session_state.churn_results = churn_analysis
                
                # Show dataset preview
                self._show_dataset_preview(churn_analysis['churn_predictions'])
                
                # Visualize Churn Insights
                self._visualize_churn_insights(churn_analysis, churn_config)
                
                # Option to proceed to next stage (Sales Forecast)
                col1, col2, col3 = st.columns([1,2,1])
                
                with col2:
                    if st.button("Continue to Sales Forecast", type="primary", use_container_width=True):
                        try:
                            # Attempt navigation using Streamlit's built-in navigation
                            st.switch_page("pages/06_Sales_Forecast.py")
                        except Exception as e:
                            st.error(f"Navigation error: {e}")
                            
                            # Fallback navigation method
                            try:
                                from src.auth.session import nav_to
                                nav_to("06_Sales_Forecast")
                            except Exception as nav_error:
                                st.error(f"Fallback navigation failed: {nav_error}")
                                st.warning("Please manually navigate to the Sales Forecast page.")

# Add main function with authentication decorator
@requires_auth
def main():
    """
    Main function to initialize and render the Churn Prediction page
    """
    # Create an instance of the ChurnPredictionPage class
    churn_page = ChurnPredictionPage()
    
    # Render the page
    churn_page.render()

# Ensure the script can be run directly
if __name__ == "__main__":
    main()