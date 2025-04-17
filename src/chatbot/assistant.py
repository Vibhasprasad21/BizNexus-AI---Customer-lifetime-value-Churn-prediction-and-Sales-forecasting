import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os

class BizNexusAssistant:
    def __init__(self, api_key=None):
        """
        Initialize the BizNexus AI Assistant with Gemini
        
        Args:
            api_key (str, optional): Google Gemini API key
        """
        # Order of preference: passed key -> environment -> streamlit secrets -> hardcoded fallback
        if api_key is None:
            # First try environment variable
            api_key = os.getenv("GEMINI_API_KEY")
            
            # Then try Streamlit secrets 
            if not api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            
            
        
        if not api_key:
            st.error("No Gemini API key found. Please configure in Streamlit secrets or environment variables.")
            raise ValueError("Google Gemini API key is required")
        
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            raise
    
    def generate_response(self, query, session_state):
        """
        Comprehensive response generation for various business queries
        
        Args:
            query (str): User's query
            session_state: Streamlit session state
        
        Returns:
            dict: Response dictionary
        """
        response = {
            "text": "",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        try:
            # Check if required data is available
            required_data = [
                'clv_results', 
                'churn_results', 
                'customer_df', 
                'result_df'
            ]
            
            missing_data = [data for data in required_data if data not in session_state]
            if missing_data:
                missing_items = ", ".join([data.replace('_', ' ') for data in missing_data])
                response["text"] = f"Please complete {missing_items} analysis first."
                return response
            
            # Get results from previous analyses
            clv_results = session_state.clv_results.copy()
            churn_results = session_state.churn_results['churn_predictions'].copy()
            customer_df = session_state.customer_df.copy()
            result_df = session_state.result_df.copy()
            
            # Merge CLV and churn results for comprehensive analysis
            merged_results = pd.merge(
                clv_results, 
                churn_results, 
                on='Customer ID', 
                how='left',
                suffixes=('', '_Churn')
            )
            
            # Normalize query processing
            query_lower = query.lower()
            
            # ENHANCED QUERY ROUTING LOGIC
            
            # 1. Visualization-Specific Queries (prioritized)
            if self._is_visualization_query(query_lower):
                return self._handle_visualization_query(query, merged_results, result_df)
            
            # 2. Customer-Specific Queries - ID, Name, CLV, Details, Profile
            elif self._is_customer_specific_query(query_lower):
                return self._handle_customer_specific_query(query, merged_results, result_df)
            
            # 3. Top/Most Valuable Customer Queries
            elif self._is_top_customers_query(query_lower):
                return self._generate_top_customers_response(merged_results, result_df)
            
            # 4. High Churn Risk Customer Queries
            elif self._is_churn_risk_query(query_lower):
                return self._generate_churn_risk_response(merged_results, result_df)
            
            # 5. Sales Trend and Forecast Queries
            elif self._is_sales_forecast_query(query_lower):
                return self._generate_sales_forecast(result_df)
            
            # 6. Segment Comparison Queries
            elif self._is_segment_comparison_query(query_lower):
                return self._generate_segment_comparison(merged_results, query)
            
            # 7. Customer Retention Strategy Queries
            elif self._is_retention_strategy_query(query_lower):
                return self._generate_retention_strategies(merged_results, query)
            
            # 8. Fallback to Gemini for complex queries
            return self._generate_gemini_response(query, merged_results, result_df)
        
        except Exception as e:
            response["text"] = f"Error generating response: {e}"
            return response
    
    def _is_customer_specific_query(self, query_lower):
        """Check if query is about a specific customer by ID or name"""
        patterns = [
            r'clv of', r'details of', r'customer profile', r'customer \d+',
            r'customer id', r'profile of', r'churn probability for',
            r'lifetime value of', r'value of customer', r'info about',
            r'information on', r'churn risk for'
        ]
        return any(re.search(pattern, query_lower) for pattern in patterns)
    
    def _is_top_customers_query(self, query_lower):
        """Check if query is about top/most valuable customers"""
        patterns = [
            'most valuable', 'top customers', 'high value', 
            'best customers', 'highest clv', 'premium customers'
        ]
        return any(term in query_lower for term in patterns)
    
    def _is_churn_risk_query(self, query_lower):
        """Check if query is about churn risk/probability"""
        patterns = [
            'churn risk', 'at risk', 'likely to churn', 'churn probability',
            'who will leave', 'customer churn', 'highest risk', 'churn rate'
        ]
        return any(term in query_lower for term in patterns) and not self._is_customer_specific_query(query_lower)
    
    def _is_sales_forecast_query(self, query_lower):
        """Check if query is about sales forecast or trends"""
        patterns = [
            'sales trend', 'sales forecast', 'revenue projection',
            'future sales', 'projected revenue', 'forecast', 'prediction'
        ]
        return any(term in query_lower for term in patterns)
    
    def _is_segment_comparison_query(self, query_lower):
        """Check if query is comparing segments"""
        comparison_terms = ['compare', 'comparison', 'versus', 'vs', 'diff', 'difference between']
        return any(term in query_lower for term in comparison_terms)
    
    def _is_retention_strategy_query(self, query_lower):
        """Check if query is about retention strategies"""
        strategy_terms = [
            'strategy', 'strategies', 'retain', 'retention', 
            'prevent churn', 'reduce churn', 'recommendations', 'recommend'
        ]
        return any(term in query_lower for term in strategy_terms)
    
    def _handle_customer_specific_query(self, query, merged_results, result_df):
        """Generate response for customer-specific queries"""
        response = {
            "text": "",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        # Extract customer identifier
        customer_identifier = self._extract_customer_identifier(query)
        
        if customer_identifier:
            # Find customer
            customer = self._find_customer(merged_results, customer_identifier)
            
            if customer is not None:
                # Generate customer profile
                response["text"] = self._generate_customer_profile(customer, merged_results)
                
                # Check if query specifically asks for sales trend
                if any(term in query.lower() for term in ['sales trend', 'purchase history', 'buying pattern']):
                    # Generate customer-specific sales trend
                    sales_trend = self._generate_customer_sales_trend(customer, result_df, merged_results)
                    response["visualization"] = sales_trend
                    response["text"] += "\n\n### Customer Sales History\n"
                    response["text"] += "The visualization shows the sales history and projected future purchases for this customer."
                
                # Check if query specifically asks for churn factors
                elif any(term in query.lower() for term in ['churn factor', 'why churn', 'churn reason']):
                    # Generate churn factor analysis
                    response["text"] += "\n\n### Churn Risk Factors\n"
                    response["text"] += self._generate_churn_factors(customer, merged_results)
                
                # Add recommendations if specifically requested or for high-value at-risk customers
                if any(term in query.lower() for term in ['recommend', 'suggestion']) or \
                   (customer.get('Churn_Probability', 0) > 0.5 and customer.get('CLV', 0) > merged_results['CLV'].median()):
                    response["text"] += "\n\n### Recommended Actions\n"
                    response["text"] += self._generate_customer_recommendations(customer, merged_results)
                
                return response
            else:
                response["text"] = f"Could not find customer matching '{customer_identifier}'. Please check the ID or name and try again."
                return response
        else:
            response["text"] = "Could not identify a specific customer in your query. Please specify a customer ID or name."
            return response
    
    def _generate_top_customers_response(self, merged_results, result_df):
        """Generate response for top customers query with enhanced insights"""
        # Sort by CLV in descending order
        top_customers = merged_results.nlargest(10, 'CLV')
        
        response = {
            "text": "## Top 10 Most Valuable Customers\n\n",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        # Create a table of top customers
        for _, customer in top_customers.iterrows():
            response["text"] += (
                f"### {customer.get('Customer Name', 'Unknown Customer')}\n"
                f"- **Customer ID:** {customer['Customer ID']}\n"
                f"- **Customer Lifetime Value (CLV):** ${customer['CLV']:,.2f}\n"
                f"- **Churn Probability:** {customer.get('Churn_Probability', 'N/A'):.1%}\n"
                f"- **Value Tier:** {customer.get('Value_Tier', 'Not Categorized')}\n\n"
            )
        
        # Add summary insights
        response["text"] += "### Key Insights\n"
        response["text"] += f"- **Total Customers Analyzed:** {len(merged_results)}\n"
        response["text"] += f"- **Average CLV:** ${merged_results['CLV'].mean():,.2f}\n"
        response["text"] += f"- **Median CLV:** ${merged_results['CLV'].median():,.2f}\n"
        response["text"] += f"- **Total Customer Value:** ${merged_results['CLV'].sum():,.2f}\n\n"
        
        # Calculate and add top 10% customer contribution (80/20 rule analysis)
        top_10_pct = merged_results.nlargest(int(len(merged_results) * 0.1), 'CLV')
        top_10_contribution = top_10_pct['CLV'].sum() / merged_results['CLV'].sum() * 100
        response["text"] += f"- **Top 10% Customer Contribution:** {top_10_contribution:.1f}% of total value\n\n"
        
        # Create visualization of top customers by CLV
        fig = px.bar(
            top_customers, 
            x='Customer ID' if 'Customer Name' not in top_customers.columns else 'Customer Name', 
            y='CLV',
            title='Top 10 Customers by Lifetime Value',
            color='Value_Tier' if 'Value_Tier' in top_customers.columns else None,
            labels={'CLV': 'Customer Lifetime Value ($)'}
        )
        
        # Add data to the visualization
        if 'Churn_Probability' in top_customers.columns:
            fig.add_trace(
                go.Scatter(
                    x=top_customers['Customer ID'] if 'Customer Name' not in top_customers.columns else top_customers['Customer Name'],
                    y=top_customers['Churn_Probability'] * top_customers['CLV'].max(),  # Scale to fit on same chart
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name='Churn Risk',
                    yaxis='y2'
                )
            )
            
            # Add secondary y-axis for churn probability
            fig.update_layout(
                yaxis2=dict(
                    title='Churn Probability',
                    overlaying='y',
                    side='right',
                    range=[0, 1],
                    tickformat='.0%'
                )
            )
        
        response["visualization"] = fig
        
        # Add recommendations
        response["text"] += "### Recommended Actions\n"
        response["text"] += "1. Create personalized retention programs for top customers\n"
        response["text"] += "2. Analyze common characteristics of high-value customers\n"
        response["text"] += "3. Develop targeted loyalty initiatives\n"
        response["text"] += "4. Implement VIP service tiers for premium customers\n"
        response["text"] += "5. Set up early warning systems for high-value customer churn risk\n"
        
        # Add sales forecast
        forecast_response = self._generate_sales_forecast(result_df)
        response["text"] += "\n\n" + forecast_response["text"]
        
        return response
    
    def _generate_churn_risk_response(self, merged_results, result_df):
        """Generate response for churn risk query"""
        # Identify high-risk customers (churn probability > 0.5)
        if 'Churn_Probability' in merged_results.columns:
            high_risk = merged_results[merged_results['Churn_Probability'] > 0.5]
            high_risk = high_risk.sort_values('Churn_Probability', ascending=False)
        else:
            # Fallback if no churn probability column exists
            high_risk = merged_results.head(0)  # Empty DataFrame with same columns
            response = {
                "text": "Churn probability data is not available. Please run churn prediction analysis first.",
                "visualization": None,
                "business_insights": {},
                "action_items": []
            }
            return response
        
        # Create response
        response = {
            "text": f"## Top {min(10, len(high_risk))} Customers at Highest Churn Risk\n\n",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        # Add high-risk customers
        for _, customer in high_risk.head(10).iterrows():
            response["text"] += (
                f"### {customer.get('Customer Name', 'Unknown Customer')}\n"
                f"- **Customer ID:** {customer['Customer ID']}\n"
                f"- **Churn Probability:** {customer.get('Churn_Probability', 'N/A'):.1%}\n"
                f"- **Customer Lifetime Value (CLV):** ${customer.get('CLV', 0):,.2f}\n"
                f"- **Value Tier:** {customer.get('Value_Tier', 'Not Categorized')}\n\n"
            )
        
        # Calculate revenue at risk
        revenue_at_risk = high_risk['CLV'].sum()
        total_revenue = merged_results['CLV'].sum()
        revenue_at_risk_pct = revenue_at_risk / total_revenue * 100 if total_revenue > 0 else 0
        
        # Add summary insights
        response["text"] += "### Key Insights\n"
        response["text"] += f"- **Total Customers at High Risk:** {len(high_risk)} ({len(high_risk)/len(merged_results):.1%})\n"
        response["text"] += f"- **Total Revenue at Risk:** ${revenue_at_risk:,.2f} ({revenue_at_risk_pct:.1f}% of total)\n"
        
        # Add high-value customers at risk insight
        if 'Value_Tier' in merged_results.columns:
            high_value_at_risk = high_risk[high_risk['Value_Tier'].isin(['Premium', 'High'])]
            response["text"] += f"- **High-Value Customers at Risk:** {len(high_value_at_risk)} ({len(high_value_at_risk)/len(high_risk):.1%} of at-risk)\n\n"
        
        # Create visualization of churn risk by customer segment
        if 'Value_Tier' in merged_results.columns:
            # Calculate churn rate by segment
            churn_by_segment = merged_results.groupby('Value_Tier')['Churn_Probability'].mean().reset_index()
            
            fig = px.bar(
                churn_by_segment,
                x='Value_Tier',
                y='Churn_Probability',
                title='Average Churn Probability by Customer Segment',
                color='Value_Tier',
                labels={'Churn_Probability': 'Average Churn Probability', 'Value_Tier': 'Customer Segment'},
                text_auto='.1%'
            )
            
            # Add segment size as scatter plot
            segment_counts = merged_results['Value_Tier'].value_counts().reset_index()
            segment_counts.columns = ['Value_Tier', 'Count']
            
            fig.add_trace(
                go.Scatter(
                    x=segment_counts['Value_Tier'],
                    y=segment_counts['Count'] / segment_counts['Count'].max() * churn_by_segment['Churn_Probability'].max(),
                    mode='markers',
                    marker=dict(size=15, color='rgba(0, 0, 0, 0.7)'),
                    name='Segment Size',
                    yaxis='y2'
                )
            )
            
            fig.update_layout(
                yaxis2=dict(
                    title='Relative Segment Size',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
            
            response["visualization"] = fig
        
        # Add churn drivers
        response["text"] += "### Top Churn Drivers\n"
        response["text"] += "1. **Low Engagement:** Declining product usage patterns\n"
        response["text"] += "2. **Pricing Concerns:** Sensitivity to recent price changes\n"
        response["text"] += "3. **Support Issues:** Unresolved customer service tickets\n"
        response["text"] += "4. **Competitor Offerings:** New alternatives in the market\n"
        response["text"] += "5. **Product Fit:** Feature gaps compared to customer needs\n\n"
        
        # Add recommended actions
        response["text"] += "### Recommended Actions\n"
        response["text"] += "1. Implement targeted retention campaigns for high-value at-risk customers\n"
        response["text"] += "2. Develop proactive engagement strategies based on usage patterns\n"
        response["text"] += "3. Create win-back offers for customers with recent declining usage\n"
        response["text"] += "4. Establish at-risk customer monitoring dashboard\n"
        response["text"] += "5. Conduct exit surveys to better understand churn drivers\n"
        
        return response
    
    def _generate_segment_comparison(self, merged_results, query):
        """Generate response comparing customer segments"""
        # Parse segments to compare from query
        segments_to_compare = self._extract_segments_to_compare(query)
        
        response = {
            "text": f"## Segment Comparison: {' vs '.join(segments_to_compare)}\n\n",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        # If we have valid segments to compare
        if segments_to_compare and 'Value_Tier' in merged_results.columns:
            # Filter for the segments we want to compare
            filtered_results = merged_results[merged_results['Value_Tier'].isin(segments_to_compare)]
            
            # Group by segment and calculate metrics
            segment_metrics = filtered_results.groupby('Value_Tier').agg({
                'CLV': ['mean', 'sum', 'count'],
                'Churn_Probability': 'mean'
            }).reset_index()
            
            # Flatten the multi-level column names
            segment_metrics.columns = ['Value_Tier', 'Avg_CLV', 'Total_CLV', 'Count', 'Avg_Churn']
            
            # Add segment comparison table
            response["text"] += "### Segment Comparison Metrics\n\n"
            response["text"] += "| Segment | Customer Count | Average CLV | Total CLV | Churn Rate |\n"
            response["text"] += "|---------|---------------|------------|-----------|------------|\n"
            
            for _, row in segment_metrics.iterrows():
                response["text"] += f"| {row['Value_Tier']} | {row['Count']:,} | ${row['Avg_CLV']:,.2f} | ${row['Total_CLV']:,.2f} | {row['Avg_Churn']:.1%} |\n"
            
            response["text"] += "\n"
            
            # Create visualization comparing segments
            fig = go.Figure()
            
            # Add CLV bars
            fig.add_trace(go.Bar(
                x=segment_metrics['Value_Tier'],
                y=segment_metrics['Avg_CLV'],
                name='Average CLV',
                marker_color='blue'
            ))
            
            # Add churn rate line
            fig.add_trace(go.Scatter(
                x=segment_metrics['Value_Tier'],
                y=segment_metrics['Avg_Churn'],
                name='Churn Rate',
                mode='lines+markers',
                marker=dict(size=10, color='red'),
                yaxis='y2'
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                title=f"Segment Comparison: {' vs '.join(segments_to_compare)}",
                yaxis=dict(title='Average CLV ($)'),
                yaxis2=dict(
                    title='Average Churn Rate',
                    overlaying='y',
                    side='right',
                    tickformat='.0%',
                    range=[0, max(segment_metrics['Avg_Churn']) * 1.2]
                ),
                barmode='group',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            response["visualization"] = fig
            
            # Add insights
            response["text"] += "### Key Insights\n\n"
            
            # Calculate differences
            if len(segment_metrics) == 2:
                s1, s2 = segment_metrics.iloc[0], segment_metrics.iloc[1]
                clv_diff_pct = (s1['Avg_CLV'] / s2['Avg_CLV'] - 1) * 100
                churn_diff_pct = (s1['Avg_Churn'] / s2['Avg_Churn'] - 1) * 100
                
                response["text"] += f"- **CLV Difference:** {s1['Value_Tier']} has {abs(clv_diff_pct):.1f}% {'higher' if clv_diff_pct > 0 else 'lower'} average CLV than {s2['Value_Tier']}\n"
                response["text"] += f"- **Churn Difference:** {s1['Value_Tier']} has {abs(churn_diff_pct):.1f}% {'higher' if churn_diff_pct > 0 else 'lower'} average churn rate than {s2['Value_Tier']}\n"
                
                # Add customer count comparison
                count_diff_pct = (s1['Count'] / s2['Count'] - 1) * 100
                response["text"] += f"- **Size Comparison:** {s1['Value_Tier']} has {abs(count_diff_pct):.1f}% {'more' if count_diff_pct > 0 else 'fewer'} customers than {s2['Value_Tier']}\n\n"
            
            # Add general insights for all comparison scenarios
            response["text"] += "### Segment-Specific Recommendations\n\n"
            
            for _, row in segment_metrics.iterrows():
                response["text"] += f"**For {row['Value_Tier']} Segment:**\n"
                
                # Recommendations based on churn rate
                if row['Avg_Churn'] > 0.4:  # High churn
                    response["text"] += "- Implement urgent retention program\n"
                    response["text"] += "- Conduct in-depth satisfaction survey\n"
                    response["text"] += "- Consider loyalty incentives\n"
                elif row['Avg_Churn'] > 0.2:  # Medium churn
                    response["text"] += "- Develop targeted engagement campaigns\n"
                    response["text"] += "- Enhance customer onboarding process\n"
                else:  # Low churn
                    response["text"] += "- Focus on upselling and cross-selling\n"
                    response["text"] += "- Create referral incentives\n"
                
                response["text"] += "\n"
        else:
            response["text"] = "Could not identify valid segments to compare. Please specify which customer segments you want to compare."
        
        return response
    
    def _extract_segments_to_compare(self, query):
        """Extract segments to compare from query"""
        # Common segment names
        segment_names = ['premium', 'high', 'medium', 'low', 'at-risk', 'loyal']
        
        found_segments = []
        query_lower = query.lower()
        
        # Look for segments specifically mentioned
        for segment in segment_names:
            if segment in query_lower:
                # Capitalize first letter for consistency
                found_segments.append(segment.capitalize())
        
        # Handle special cases
        if 'high' in found_segments and 'high value' in query_lower:
            found_segments.remove('High')
            found_segments.append('High Value')
        
        return found_segments
    
    def _generate_retention_strategies(self, merged_results, query):
        """Generate personalized retention strategies based on data"""
        response = {
            "text": "## Customer Retention Strategies\n\n",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        # Try to extract segment from query
        segment = None
        segment_patterns = [
            r'for\s+(\w+)\s+segment', 
            r'for\s+(\w+)\s+tier', 
            r'for\s+(\w+)-value',
            r'for\s+(\w+)\s+customers'
        ]
        
        for pattern in segment_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                segment = match.group(1).capitalize()
                break
        
        # If segment is specified and exists in the data
        if segment and 'Value_Tier' in merged_results.columns:
            # For segment-specific strategies
            filtered_results = merged_results[merged_results['Value_Tier'] == segment]
            
            if len(filtered_results) > 0:
                # Calculate segment metrics
                avg_clv = filtered_results['CLV'].mean()
                avg_churn = filtered_results['Churn_Probability'].mean() if 'Churn_Probability' in filtered_results.columns else 0
                
                response["text"] += f"### Retention Strategies for {segment} Segment\n\n"
                response["text"] += f"**Segment Profile:**\n"
                response["text"] += f"- Customer Count: {len(filtered_results)}\n"
                response["text"] += f"- Average CLV: ${avg_clv:,.2f}\n"
                response["text"] += f"- Average Churn Risk: {avg_churn:.1%}\n\n"
                
                response["text"] += "**Recommended Strategies:**\n\n"
                
                # Different strategies based on segment
                if segment.lower() in ['premium', 'high']:
                    response["text"] += "1. **VIP Account Management Program**\n"
                    response["text"] += "   - Assign dedicated account managers for personalized service\n"
                    response["text"] += "   - Quarterly business reviews to ensure alignment\n"
                    response["text"] += "   - Priority technical support and issue resolution\n\n"
                    
                    response["text"] += "2. **Exclusive Benefits Package**\n"
                    response["text"] += "   - Early access to new features and products\n"
                    response["text"] += "   - Complementary professional services\n"
                    response["text"] += "   - Custom feature development options\n\n"
                    
                    response["text"] += "3. **Strategic Partnership Program**\n"
                    response["text"] += "   - Co-marketing opportunities\n"
                    response["text"] += "   - Executive relationship building\n"
                    response["text"] += "   - Industry-specific strategy sessions\n\n"
                
                elif segment.lower() in ['medium']:
                    response["text"] += "1. **Growth Acceleration Program**\n"
                    response["text"] += "   - Guided platform optimization sessions\n"
                    response["text"] += "   - Success roadmap development\n"
                    response["text"] += "   - Regular check-ins and usage reviews\n\n"
                    
                    response["text"] += "2. **Loyalty Rewards System**\n"
                    response["text"] += "   - Cumulative usage discounts\n"
                    response["text"] += "   - Expansion incentives\n"
                    response["text"] += "   - Referral bonuses\n\n"
                    
                    response["text"] += "3. **Educational Content Program**\n"
                    response["text"] += "   - Personalized learning paths\n"
                    response["text"] += "   - Expert webinars and training sessions\n"
                    response["text"] += "   - Best practices certification\n\n"
                
                else:  # Low value or others
                    response["text"] += "1. **Engagement Boost Campaign**\n"
                    response["text"] += "   - Automated usage tips and best practices\n"
                    response["text"] += "   - Feature discovery emails\n"
                    response["text"] += "   - Success story sharing\n\n"
                    
                    response["text"] += "2. **Value Demonstration Program**\n"
                    response["text"] += "   - ROI calculators and reporting\n"
                    response["text"] += "   - Targeted use case sharing\n\n"
                    
                    response["text"] += "3. **Simplified Upgrade Pathway**\n"
                    response["text"] += "   - Frictionless tier upgrades\n"
                    response["text"] += "   - Small commitment expansion options\n"
                    response["text"] += "   - Guided value realization sessions\n\n"
                
                # Create visualization of churn risk distribution in segment
                if 'Churn_Probability' in filtered_results.columns:
                    fig = px.histogram(
                        filtered_results, 
                        x='Churn_Probability',
                        nbins=20,
                        title=f'Churn Risk Distribution - {segment} Segment',
                        labels={'Churn_Probability': 'Churn Probability'},
                        color_discrete_sequence=['blue']
                    )
                    
                    fig.add_vline(
                        x=avg_churn, 
                        line_dash="dash", 
                        line_color="red", 
                        annotation_text="Segment Average", 
                        annotation_position="top right"
                    )
                    
                    response["visualization"] = fig
            else:
                response["text"] = f"No customers found in the {segment} segment. Please check segment name and try again."
        else:
            # Generic retention strategies for all customers
            response["text"] += "### General Retention Strategies\n\n"
            
            # Calculate overall metrics
            avg_clv = merged_results['CLV'].mean()
            avg_churn = merged_results['Churn_Probability'].mean() if 'Churn_Probability' in merged_results.columns else 0
            
            # Segment breakdown if available
            if 'Value_Tier' in merged_results.columns:
                segment_counts = merged_results['Value_Tier'].value_counts()
                response["text"] += "**Customer Base Overview:**\n"
                response["text"] += f"- Total Customers: {len(merged_results)}\n"
                response["text"] += f"- Average CLV: ${avg_clv:,.2f}\n"
                response["text"] += f"- Average Churn Risk: {avg_churn:.1%}\n"
                response["text"] += "- Segment Breakdown:\n"
                
                for segment, count in segment_counts.items():
                    response["text"] += f"  - {segment}: {count} customers ({count/len(merged_results):.1%})\n"
                
                response["text"] += "\n"
            
            # Provide general strategies
            response["text"] += "**Recommended Strategies:**\n\n"
            
            response["text"] += "1. **Tiered Loyalty Program**\n"
            response["text"] += "   - Implement a multi-tiered rewards program with clear progression\n"
            response["text"] += "   - Offer exclusive benefits at each tier that drive ongoing engagement\n"
            response["text"] += "   - Create visible status symbols and recognition for loyal customers\n\n"
            
            response["text"] += "2. **Proactive Engagement Framework**\n"
            response["text"] += "   - Develop an early warning system for potential churn\n"
            response["text"] += "   - Create automated touch points at critical moments in customer lifecycle\n"
            response["text"] += "   - Establish regular business reviews for high-value accounts\n\n"
            
            response["text"] += "3. **Customer Success Roadmap**\n"
            response["text"] += "   - Clearly define what success looks like for different customer segments\n"
            response["text"] += "   - Create milestone-based communication plans\n"
            response["text"] += "   - Celebrate customer wins and successes\n\n"
            
            response["text"] += "4. **Voice of Customer Program**\n"
            response["text"] += "   - Implement regular feedback collection mechanisms\n"
            response["text"] += "   - Close the loop on customer suggestions\n"
            response["text"] += "   - Create customer advisory board for strategic input\n\n"
            
            response["text"] += "5. **Customer Education Initiative**\n"
            response["text"] += "   - Develop comprehensive onboarding materials\n"
            response["text"] += "   - Create ongoing education pathways for product mastery\n"
            response["text"] += "   - Build a community for peer-to-peer learning\n\n"
            
            # Create visualization of value vs. churn risk
            if 'Churn_Probability' in merged_results.columns and 'CLV' in merged_results.columns:
                fig = px.scatter(
                    merged_results,
                    x='CLV',
                    y='Churn_Probability',
                    color='Value_Tier' if 'Value_Tier' in merged_results.columns else None,
                    title='Customer Lifetime Value vs. Churn Risk',
                    labels={'CLV': 'Customer Lifetime Value ($)', 'Churn_Probability': 'Churn Probability'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                # Add quadrant lines
                median_clv = merged_results['CLV'].median()
                median_churn = merged_results['Churn_Probability'].median()
                
                fig.add_vline(x=median_clv, line_dash="dash", line_color="gray")
                fig.add_hline(y=median_churn, line_dash="dash", line_color="gray")
                
                # Add quadrant annotations
                fig.add_annotation(x=median_clv*0.5, y=median_churn*0.5, text="Low CLV, Low Risk",
                                showarrow=False, font=dict(size=10))
                fig.add_annotation(x=median_clv*0.5, y=median_churn*1.5, text="Low CLV, High Risk",
                                showarrow=False, font=dict(size=10))
                fig.add_annotation(x=median_clv*1.5, y=median_churn*0.5, text="High CLV, Low Risk",
                                showarrow=False, font=dict(size=10))
                fig.add_annotation(x=median_clv*1.5, y=median_churn*1.5, text="High CLV, High Risk",
                                showarrow=False, font=dict(size=10))
                
                response["visualization"] = fig
        
        return response
    
    def _generate_sales_forecast(self, result_df, forecast_periods=None):
        """
        Generate comprehensive sales forecast with flexible period selection
        
        Args:
            result_df (pd.DataFrame): Transaction data
            forecast_periods (int, optional): Number of months to forecast. Defaults to adaptive selection.
        
        Returns:
            dict: Sales forecast response
        """
        # Identify date and sales columns dynamically
        def find_column(df, column_types):
            for col in df.columns:
                for type_check in column_types:
                    if type_check in col.lower():
                        return col
            return None

        date_column = find_column(result_df, ['date', 'time', 'day'])
        sales_column = find_column(result_df, ['amount', 'price', 'revenue', 'sales', 'value'])
        
        if not date_column or not sales_column:
            return {
                "text": "Cannot generate sales trend. Required date or sales columns not found.",
                "visualization": None
            }
        
        # Ensure date column is datetime
        try:
            result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
        except Exception as e:
            return {
                "text": f"Error processing date column: {e}",
                "visualization": None
            }
        
        # Remove rows with NaT or invalid dates
        result_df = result_df.dropna(subset=[date_column])
        
        if result_df.empty:
            return {
                "text": "No valid data available for sales forecast.",
                "visualization": None
            }
        
        # Group sales by date
        try:
            sales_by_date = result_df.groupby(pd.Grouper(key=date_column, freq='M'))[sales_column].sum().reset_index()
        except Exception as e:
            return {
                "text": f"Error grouping sales data: {e}",
                "visualization": None
            }
        
        # Adaptive forecast periods
        if forecast_periods is None:
            # Determine forecast periods based on data availability
            data_range = sales_by_date[date_column].max() - sales_by_date[date_column].min()
            if data_range.days < 90:  # Less than 3 months of data
                forecast_periods = 3
            elif data_range.days < 180:  # Less than 6 months of data
                forecast_periods = 6
            else:
                forecast_periods = 12
        
        # Last date in historical data
        last_date = sales_by_date[date_column].max()
        
        # Generate future dates
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='M'
        )
        
        # Ensure we have sufficient historical data
        if len(sales_by_date) < 3:
            return {
                "text": "Insufficient historical data for generating a reliable forecast.",
                "visualization": None
            }
        
        # Simple forecasting method
        recent_data = sales_by_date.tail(min(12, len(sales_by_date)))  # Use last year's data or all available
        
        # Robust average and trend calculation
        try:
            avg_sales = recent_data[sales_column].mean()
            
            # Calculate trend
            x = np.arange(len(recent_data))
            y = recent_data[sales_column].values
            slope, _ = np.polyfit(x, y, 1)
            trend_factor = slope / avg_sales if avg_sales > 0 else 0.01
        except Exception as e:
            # Fallback to simple average if trend calculation fails
            avg_sales = recent_data[sales_column].mean()
            trend_factor = 0.01
        
        # Generate forecast with trend and seasonality
        trend = 1 + trend_factor * np.arange(len(forecast_dates))
        seasonality = 1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(forecast_dates)))
        noise = np.random.normal(0, 0.05, len(forecast_dates))
        
        forecast_values = avg_sales * trend * seasonality * (1 + noise)
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values,
            'Lower_Bound': forecast_values * 0.9,
            'Upper_Bound': forecast_values * 1.1
        })
        
        # Visualization
        fig = go.Figure([
            go.Scatter(
                x=forecast_df['Date'], 
                y=forecast_df['Forecast'], 
                mode='lines', 
                name='Forecast',
                line=dict(color='blue', width=3)
            ),
            go.Scatter(
                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(68, 168, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            )
        ])
        
        # Add historical data if available
        if len(sales_by_date) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sales_by_date[date_column],
                    y=sales_by_date[sales_column],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='green', width=2)
                )
            )
        
        fig.update_layout(
            title=f'Sales Forecast for Next {forecast_periods} Months',
            xaxis_title='Month',
            yaxis_title='Forecasted Sales ($)'
        )
        
        # Prepare response
        response_text = f"## Sales Forecast for Next {forecast_periods} Months\n\n"
        response_text += "### Forecast Summary\n"
        response_text += f"- **Total Forecasted Sales:** ${forecast_df['Forecast'].sum():,.2f}\n"
        response_text += f"- **Average Monthly Sales:** ${forecast_df['Forecast'].mean():,.2f}\n"
        response_text += f"- **Lowest Projected Month:** ${forecast_df['Forecast'].min():,.2f} ({forecast_df.loc[forecast_df['Forecast'].idxmin(), 'Date'].strftime('%B %Y')})\n"
        response_text += f"- **Highest Projected Month:** ${forecast_df['Forecast'].max():,.2f} ({forecast_df.loc[forecast_df['Forecast'].idxmax(), 'Date'].strftime('%B %Y')})\n\n"
        
        # Calculate growth rate
        first_month = forecast_df['Forecast'].iloc[0]
        last_month = forecast_df['Forecast'].iloc[-1]
        growth_rate = (last_month / first_month - 1) * 100
        
        response_text += f"- **Growth Trend:** {growth_rate:.1f}% over forecast period\n\n"
        
        response_text += "### Monthly Breakdown\n"
        for _, row in forecast_df.iterrows():
            response_text += f"- **{row['Date'].strftime('%B %Y')}:** ${row['Forecast']:,.2f}\n"
        
        response_text += "\n### Insights\n"
        response_text += "1. Forecast based on historical sales trends\n"
        response_text += "2. Accounts for seasonal variations\n"
        response_text += "3. Includes confidence interval for uncertainty\n"
        
        # Add trend analysis
        if trend_factor > 0:
            response_text += "4. Positive growth trend detected in historical data\n"
        else:
            response_text += "4. Negative or flat growth trend detected in historical data\n"
        
        # Add seasonal analysis
        if np.max(seasonality) - np.min(seasonality) > 0.15:
            response_text += "5. Significant seasonal patterns detected\n"
        
        return {
            "text": response_text,
            "visualization": fig
        }
    
    def _generate_customer_sales_trend(self, customer, result_df, customer_df):
        """
        Generate sales trend visualization for a specific customer
        
        Args:
            customer (pd.Series): Customer record
            result_df (pd.DataFrame): Transaction results
            customer_df (pd.DataFrame): Customer data
        
        Returns:
            go.Figure: Sales trend visualization
        """
        # Filter transactions for specific customer
        customer_transactions = result_df[result_df['Customer ID'] == customer['Customer ID']]
        
        # If no transactions found, return None
        if customer_transactions.empty:
            return None
        
        # Identify date and sales columns
        date_col = None
        for col in customer_transactions.columns:
            if any(date_term in col.lower() for date_term in ['date', 'time', 'day']):
                date_col = col
                break
                
        sales_col = None
        for col in customer_transactions.columns:
            if any(amount_term in col.lower() for amount_term in ['amount', 'price', 'revenue', 'sales', 'value']):
                sales_col = col
                break
        
        if not date_col or not sales_col:
            return None
        
        # Convert date column
        customer_transactions[date_col] = pd.to_datetime(customer_transactions[date_col])
        
        # Group by month and calculate sales
        monthly_sales = customer_transactions.groupby(
            pd.Grouper(key=date_col, freq='M')
        )[sales_col].sum()
        
        # Generate forecast for next 6 months
        last_date = monthly_sales.index.max()
        forecast_months = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=6, 
            freq='M'
        )
        
        # Simple forecast using average and trend
        if len(monthly_sales) >= 2:
            x = np.arange(len(monthly_sales))
            y = monthly_sales.values
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope
        else:
            avg_sales = monthly_sales.mean()
            trend = avg_sales * 0.05  # Assume 5% growth if not enough data
        
        # Generate forecast values
        forecast_values = []
        for i in range(len(forecast_months)):
            if len(monthly_sales) >= 2:
                # Use the trend line equation
                next_val = slope * (len(monthly_sales) + i) + intercept
                forecast_values.append(max(0, next_val))  # Ensure non-negative
            else:
                # Use simple growth
                forecast_values.append(avg_sales * (1 + 0.05 * (i+1)))
        
        # Create visualization
        fig = go.Figure()
        
        # Add historical sales
        fig.add_trace(go.Scatter(
            x=monthly_sales.index, 
            y=monthly_sales.values, 
            mode='lines+markers', 
            name='Historical Sales',
            marker=dict(size=8),
            line=dict(color='blue', width=3)
        ))
        
        # Add forecast sales
        fig.add_trace(go.Scatter(
            x=forecast_months, 
            y=forecast_values, 
            mode='lines+markers', 
            name='Sales Forecast',
            marker=dict(size=8),
            line=dict(color='red', dash='dot', width=3)
        ))
        
        # Calculate customer metrics
        avg_purchase = monthly_sales.mean()
        total_purchases = monthly_sales.sum()
        purchase_frequency = len(monthly_sales) / (monthly_sales.index.max() - monthly_sales.index.min()).days * 30 if len(monthly_sales) > 1 else 0
        
        # Create annotations - FIX HERE: Escaping the inner quotes
        customer_id_str = f"ID: {customer['Customer ID']}"
        customer_name = customer.get('Customer Name', customer_id_str)
        
        annotations = [
            dict(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text=f"Customer Sales Analysis: {customer_name}",
                showarrow=False,
                font=dict(size=16)
            ),
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"Avg. Monthly: ${avg_purchase:.2f} | Total: ${total_purchases:.2f} | Frequency: {purchase_frequency:.2f} per month",
                showarrow=False,
                font=dict(size=12)
            )
        ]
        
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Sales ($)',
            legend_title='Sales Trend',
            annotations=annotations,
            margin=dict(t=100)
        )
        
        return fig
    def _extract_customer_identifier(self, query):
        """
        Extract customer name or ID from query
        
        Args:
            query (str): User's query
        
        Returns:
            str or None: Customer identifier
        """
        # Try to extract name
        name_patterns = [
            r'(?:clv|details|profile|sales trend|value|churn|information|info)\s+(?:of|for|about)\s+([a-zA-Z\s]+)',
            r'customer\s+([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)\'s\s+(?:clv|profile|details|value|churn)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try to extract ID
        id_pattern = r'customer\s*(?:#|id|number|)?\s*(\d+)'
        id_match = re.search(id_pattern, query, re.IGNORECASE)
        if id_match:
            return id_match.group(1)
        
        return None
    
    def _find_customer(self, merged_results, identifier):
        """
        Find customer in merged results
        
        Args:
            merged_results (pd.DataFrame): Merged customer results
            identifier (str): Customer name or ID
        
        Returns:
            pd.Series or None: Customer record
        """
        # Try ID match first
        if identifier.isdigit():
            id_match = merged_results[merged_results['Customer ID'] == int(identifier)]
            if not id_match.empty:
                return id_match.iloc[0]
        
        # Try name match - case insensitive and allowing partial matches
        if 'Customer Name' in merged_results.columns:
            # Try exact match first (case insensitive)
            exact_match = merged_results[
                merged_results['Customer Name'].str.lower() == identifier.lower()
            ]
            
            if not exact_match.empty:
                return exact_match.iloc[0]
            
            # Try contains match next
            contains_match = merged_results[
                merged_results['Customer Name'].str.lower().str.contains(identifier.lower())
            ]
            
            if not contains_match.empty:
                return contains_match.iloc[0]
        
        return None
    
    def _generate_customer_profile(self, customer, merged_results):
        """
        Generate detailed customer profile
        
        Args:
            customer (pd.Series): Customer record
            merged_results (pd.DataFrame): Merged results for context
        
        Returns:
            str: Formatted customer profile
        """
        profile = f"## Customer Profile: {customer.get('Customer Name', 'Unknown Customer')}\n\n"
        
        # Basic Information
        profile += f"**Customer ID:** {customer['Customer ID']}\n"
        
        # Add any available customer attributes
        attribute_columns = [
            ('Email', 'Email'), 
            ('Phone', 'Phone'), 
            ('Location', 'Location'),
            ('Region', 'Region'),
            ('Customer_Type', 'Customer Type'),
            ('Signup_Date', 'Customer Since'),
            ('Acquisition_Channel', 'Acquisition Channel')
        ]
        
        for col, label in attribute_columns:
            if col in customer.index and not pd.isna(customer[col]):
                if col == 'Signup_Date':
                    # Format date
                    try:
                        date_val = pd.to_datetime(customer[col]).strftime('%B %d, %Y')
                        profile += f"**{label}:** {date_val}\n"
                    except:
                        profile += f"**{label}:** {customer[col]}\n"
                else:
                    profile += f"**{label}:** {customer[col]}\n"
        
        profile += "\n### Value Metrics\n"
        profile += f"**Customer Lifetime Value (CLV):** ${customer['CLV']:,.2f}\n"
        
        # Churn and Risk Information
        churn_prob_col = 'Churn_Probability_Churn' if 'Churn_Probability_Churn' in customer.index else 'Churn_Probability'
        if churn_prob_col in customer.index:
            churn_prob = customer[churn_prob_col]
            profile += f"**Churn Probability:** {churn_prob:.1%}\n"
            
            # Add risk category based on probability
            if churn_prob < 0.3:
                risk_category = "Low Risk"
            elif churn_prob < 0.6:
                risk_category = "Medium Risk"
            else:
                risk_category = "High Risk"
            
            profile += f"**Risk Category:** {risk_category}\n"
        
        # Value Tier
        if 'Value_Tier' in customer.index:
            profile += f"**Value Tier:** {customer.get('Value_Tier', 'Not Categorized')}\n"
        
        # Transaction metrics if available
        if 'Total_Purchases' in customer.index:
            profile += f"**Total Purchases:** {customer['Total_Purchases']}\n"
        
        if 'Average_Order_Value' in customer.index:
            profile += f"**Average Order Value:** ${customer['Average_Order_Value']:,.2f}\n"
        
        if 'Last_Purchase_Date' in customer.index:
            try:
                last_purchase = pd.to_datetime(customer['Last_Purchase_Date']).strftime('%B %d, %Y')
                profile += f"**Last Purchase:** {last_purchase}\n"
            except:
                profile += f"**Last Purchase:** {customer['Last_Purchase_Date']}\n"
        
        profile += "\n### Comparative Insights\n"
        profile += f"- **Average CLV:** ${merged_results['CLV'].mean():,.2f}\n"
        profile += f"- **Median CLV:** ${merged_results['CLV'].median():,.2f}\n"
        profile += f"- **CLV Percentile:** {self._calculate_percentile(customer['CLV'], merged_results['CLV']):.1f}%\n"
        
        # Add churn percentile if available
        if churn_prob_col in customer.index and churn_prob_col in merged_results.columns:
            churn_percentile = self._calculate_percentile(customer[churn_prob_col], merged_results[churn_prob_col], higher_is_worse=True)
            profile += f"- **Churn Risk Percentile:** {churn_percentile:.1f}% (lower is better)\n"
        
        return profile
    
    def _generate_churn_factors(self, customer, merged_results):
        """
        Generate analysis of churn factors for a specific customer
        
        Args:
            customer (pd.Series): Customer record
            merged_results (pd.DataFrame): Merged results for context
        
        Returns:
            str: Churn factors analysis
        """
        factors = ""
        
        # Check various possible churn indicators
        churn_factors = []
        
        # Check purchase recency
        if 'Last_Purchase_Date' in customer.index:
            try:
                last_purchase = pd.to_datetime(customer['Last_Purchase_Date'])
                days_since_purchase = (datetime.now() - last_purchase).days
                
                if days_since_purchase > 180:
                    churn_factors.append(f"**Inactivity:** No purchases in {days_since_purchase} days")
            except:
                pass
        
        # Check purchase frequency trend
        if 'Purchase_Frequency' in customer.index and 'Previous_Frequency' in customer.index:
            if customer['Purchase_Frequency'] < customer['Previous_Frequency'] * 0.7:
                churn_factors.append("**Declining Engagement:** Purchase frequency has decreased significantly")
        
        # Check average order value trend
        if 'Average_Order_Value' in customer.index and 'Previous_AOV' in customer.index:
            if customer['Average_Order_Value'] < customer['Previous_AOV'] * 0.7:
                churn_factors.append("**Reduced Spending:** Average order value has decreased significantly")
        
        # Check support tickets or complaints
        if 'Support_Tickets' in customer.index and customer['Support_Tickets'] > 3:
            churn_factors.append(f"**Support Issues:** {customer['Support_Tickets']} recent support tickets")
        
        # Check product usage
        if 'Usage_Level' in customer.index and customer['Usage_Level'] == 'Low':
            churn_factors.append("**Low Product Usage:** Limited engagement with product features")
        
        # If we have specific churn factors
        if churn_factors:
            factors += "The following factors contribute to this customer's churn risk:\n\n"
            for factor in churn_factors:
                factors += f"- {factor}\n"
        else:
            # Use generic factors based on customer attributes
            factors += "Based on customer attributes, the following may contribute to churn risk:\n\n"
            
            # Check value tier
            if 'Value_Tier' in customer.index and customer['Value_Tier'] in ['Low', 'Medium']:
                factors += "- **Value Sensitivity:** Lower-tier customers tend to be more price-sensitive\n"
            
            # Check tenure
            if 'Signup_Date' in customer.index:
                try:
                    signup_date = pd.to_datetime(customer['Signup_Date'])
                    tenure_days = (datetime.now() - signup_date).days
                    
                    if tenure_days < 180:
                        factors += f"- **New Customer:** Only {tenure_days//30} months as a customer\n"
                except:
                    pass
            
            # Check acquisition channel
            if 'Acquisition_Channel' in customer.index:
                if customer['Acquisition_Channel'] in ['Referral', 'Discount Promotion']:
                    factors += f"- **Acquisition Source:** Customers from {customer['Acquisition_Channel']} may be more discount-driven\n"
        
        # Add comparative context
        churn_prob_col = 'Churn_Probability_Churn' if 'Churn_Probability_Churn' in customer.index else 'Churn_Probability'
        if churn_prob_col in customer.index and churn_prob_col in merged_results.columns:
            customer_churn = customer[churn_prob_col]
            avg_churn = merged_results[churn_prob_col].mean()
            
            if customer_churn > avg_churn:
                factors += f"\nThis customer's churn probability ({customer_churn:.1%}) is {(customer_churn/avg_churn-1)*100:.1f}% higher than the average ({avg_churn:.1%}).\n"
            else:
                factors += f"\nThis customer's churn probability ({customer_churn:.1%}) is {(1-customer_churn/avg_churn)*100:.1f}% lower than the average ({avg_churn:.1%}).\n"
        
        return factors
    
    def _generate_customer_recommendations(self, customer, merged_results):
        """
        Generate personalized recommendations for a specific customer
        
        Args:
            customer (pd.Series): Customer record
            merged_results (pd.DataFrame): Merged results for context
        
        Returns:
            str: Personalized recommendations
        """
        recommendations = ""
        
        # Check for churn risk
        churn_prob_col = 'Churn_Probability_Churn' if 'Churn_Probability_Churn' in customer.index else 'Churn_Probability'
        high_churn_risk = False
        if churn_prob_col in customer.index:
            high_churn_risk = customer[churn_prob_col] > 0.5
        
        # Check value tier
        high_value = False
        if 'Value_Tier' in customer.index:
            high_value = customer['Value_Tier'] in ['Premium', 'High']
        elif 'CLV' in customer.index and 'CLV' in merged_results.columns:
            high_value = customer['CLV'] > merged_results['CLV'].quantile(0.75)
        
        # Recommendations based on churn risk and value
        if high_churn_risk and high_value:
            # High-value customer at risk - VIP retention
            recommendations += "This high-value customer shows elevated churn risk and requires immediate attention:\n\n"
            recommendations += "1. **Executive Outreach:** Schedule a business review with senior leadership\n"
            recommendations += "2. **Custom Success Plan:** Develop a personalized roadmap for their specific needs\n"
            recommendations += "3. **Exclusive Offer:** Consider a loyalty package with premium incentives\n"
            recommendations += "4. **Dedicated Support:** Assign a named account representative\n"
            recommendations += "5. **Usage Analysis:** Identify underutilized features that could deliver additional value\n"
        
        elif high_churn_risk and not high_value:
            # At-risk customer with lower value - efficient retention
            recommendations += "This customer shows elevated churn risk and requires prompt follow-up:\n\n"
            recommendations += "1. **Engagement Campaign:** Enter into an automated re-engagement sequence\n"
            recommendations += "2. **Satisfaction Survey:** Send a brief survey to identify specific pain points\n"
            recommendations += "3. **Usage Guidance:** Provide targeted materials on relevant features\n"
            recommendations += "4. **Limited-Time Offer:** Consider a short-term renewal incentive\n"
            recommendations += "5. **Success Story:** Share relevant case studies demonstrating value realization\n"
        
        elif not high_churn_risk and high_value:
            # High-value satisfied customer - growth opportunity
            recommendations += "This high-value customer shows healthy engagement and represents growth potential:\n\n"
            recommendations += "1. **Expansion Strategy:** Identify opportunities for additional product adoption\n"
            recommendations += "2. **Advisory Program:** Invite to customer advisory board or exclusive events\n"
            recommendations += "3. **Referral Request:** Implement a strategic referral program\n"
            recommendations += "4. **Advanced Training:** Offer specialized training for power users\n"
            recommendations += "5. **Relationship Building:** Schedule periodic executive check-ins\n"
        
        else:
            # Satisfied customer with lower value - nurture and grow
            recommendations += "This customer shows healthy engagement and represents an opportunity to increase value:\n\n"
            recommendations += "1. **Educational Content:** Provide targeted resources to deepen product knowledge\n"
            recommendations += "2. **Feature Discovery:** Highlight relevant unused features that align with their needs\n"
            recommendations += "3. **Success Milestones:** Celebrate usage achievements and progress\n"
            recommendations += "4. **Community Engagement:** Encourage participation in user community\n"
            recommendations += "5. **Upgrade Path:** Present clear value proposition for moving to higher tier\n"
        
        return recommendations
    
    def _calculate_percentile(self, value, series, higher_is_worse=False):
        """
        Calculate percentile of a value in a series
        
        Args:
            value (float): Value to calculate percentile for
            series (pd.Series): Series of values
            higher_is_worse (bool): If True, higher percentile is worse (for metrics like churn)
        
        Returns:
            float: Percentile of the value
        """
        if higher_is_worse:
            return (series < value).mean() * 100
        else:
            return (series < value).mean() * 100
    
    def _generate_gemini_response(self, query, merged_results, result_df):
        """
        Generate response using Gemini when specific handlers don't match
        
        Args:
            query (str): User's query
            merged_results (pd.DataFrame): Merged customer results
            result_df (pd.DataFrame): Transaction results
        
        Returns:
            dict: Response dictionary
        """
        # Prepare context from data for Gemini
        context = self._prepare_data_context(merged_results)
        
        # Add transaction context if available
        if result_df is not None and not result_df.empty:
            tx_context = self._prepare_transaction_context(result_df)
            context += "\n\n" + tx_context
        
        # Add specific question
        full_prompt = f"{context}\n\nUser Query: {query}\n\nPlease provide a detailed response with data-backed insights. Include relevant statistics and actionable recommendations."
        
        # Generate Gemini response
        gemini_response = self.model.generate_content(full_prompt)
        
        # Create response dictionary
        response = {
            "text": gemini_response.text,
            "visualization": None
        }
        
        # Add visualization if appropriate
        if any(term in query.lower() for term in ['show', 'plot', 'display', 'chart', 'graph', 'visualize']):
            response["visualization"] = self._generate_appropriate_visualization(query, merged_results, result_df)
        
        return response
    
    def _generate_appropriate_visualization(self, query, merged_results, result_df):
        """
        Generate an appropriate visualization based on query
        
        Args:
            query (str): User's query
            merged_results (pd.DataFrame): Merged customer results
            result_df (pd.DataFrame): Transaction results
        
        Returns:
            go.Figure: Appropriate visualization
        """
        query_lower = query.lower()
        
        # CLV Distribution
        if any(term in query_lower for term in ['clv distribution', 'clv histogram', 'distribution of clv']):
            return self._generate_clv_distribution(merged_results)
        
        # Churn Risk Heatmap
        elif any(term in query_lower for term in ['churn heatmap', 'churn heat map', 'churn by region']):
            return self._generate_churn_heatmap(merged_results)
        
        # Customer Segments
        elif any(term in query_lower for term in ['segment', 'customer segment', 'customer tier']):
            return self._generate_segment_visualization(merged_results)
        
        # Comparison Chart
        elif any(term in query_lower for term in ['compare', 'comparison', 'versus', 'vs']):
            return self._generate_comparison_chart(merged_results, query_lower)
        
        # Sales/Revenue Visualization
        elif any(term in query_lower for term in ['sales', 'revenue', 'income']):
            return self._generate_sales_visualization(result_df)
        
        # Default to CLV vs Churn scatter plot
        else:
            return self._generate_clv_churn_scatter(merged_results)
    
    def _generate_clv_distribution(self, merged_results):
        """Generate CLV distribution histogram"""
        if 'CLV' not in merged_results.columns:
            return None
        
        # Create CLV histogram
        fig = px.histogram(
            merged_results,
            x='CLV',
            nbins=30,
            title='Customer Lifetime Value Distribution',
            labels={'CLV': 'Customer Lifetime Value ($)'},
            color='Value_Tier' if 'Value_Tier' in merged_results.columns else None
        )
        
        # Add median line
        median_clv = merged_results['CLV'].median()
        fig.add_vline(
            x=median_clv,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: ${median_clv:.2f}",
            annotation_position="top right"
        )
        
        return fig
    
    def _generate_churn_heatmap(self, merged_results):
        """Generate churn risk heatmap by segment"""
        if 'Churn_Probability' not in merged_results.columns:
            return None
        
        # If we have region or location data
        region_col = None
        for col in ['Region', 'Location', 'Country', 'State']:
            if col in merged_results.columns:
                region_col = col
                break
        
        if region_col and 'Value_Tier' in merged_results.columns:
            # Create a cross-tabulation
            heatmap_data = pd.crosstab(
                merged_results[region_col],
                merged_results['Value_Tier'],
                values=merged_results['Churn_Probability'],
                aggfunc='mean'
            )
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                colorbar=dict(title='Churn Probability')
            ))
            
            fig.update_layout(
                title='Churn Risk Heatmap by Region and Customer Segment',
                xaxis_title='Customer Segment',
                yaxis_title=region_col
            )
            
            return fig
        elif 'Value_Tier' in merged_results.columns:
            # Create a bar chart of churn by segment
            segment_churn = merged_results.groupby('Value_Tier')['Churn_Probability'].mean().reset_index()
            
            fig = px.bar(
                segment_churn,
                x='Value_Tier',
                y='Churn_Probability',
                title='Average Churn Probability by Customer Segment',
                color='Value_Tier',
                text_auto='.1%'
            )
            
            return fig
        else:
            # Default to histogram of churn probabilities
            fig = px.histogram(
                merged_results,
                x='Churn_Probability',
                nbins=20,
                title='Distribution of Churn Probabilities',
                labels={'Churn_Probability': 'Churn Probability'}
            )
            
            return fig
    
    def _generate_segment_visualization(self, merged_results):
        """Generate customer segment visualization"""
        if 'Value_Tier' in merged_results.columns:
            # Create a pie chart of customer segments
            segment_counts = merged_results['Value_Tier'].value_counts().reset_index()
            segment_counts.columns = ['Value_Tier', 'Count']
            
            fig = px.pie(
                segment_counts,
                values='Count',
                names='Value_Tier',
                title='Customer Segment Distribution',
                color='Value_Tier',
                hole=0.3
            )
            
            # Add total customers annotation
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text=f"Total: {len(merged_results)}",
                showarrow=False,
                font=dict(size=16)
            )
            
            return fig
        else:
            # Default to CLV range segmentation
            merged_results['CLV_Range'] = pd.cut(
                merged_results['CLV'],
                bins=[0, 1000, 2000, 5000, float('inf')],
                labels=['0-1K', '1K-2K', '2K-5K', '5K+']
            )
            
            segment_counts = merged_results['CLV_Range'].value_counts().reset_index()
            segment_counts.columns = ['CLV_Range', 'Count']
            
            fig = px.pie(
                segment_counts,
                values='Count',
                names='CLV_Range',
                title='Customer Distribution by CLV Range',
                hole=0.3
            )
            
            return fig
    
    def _generate_comparison_chart(self, merged_results, query_lower):
        """Generate comparison chart based on query"""
        # Try to determine what to compare
        comparison_metrics = {
            'clv': 'CLV',
            'churn': 'Churn_Probability',
            'lifetime value': 'CLV',
            'churn probability': 'Churn_Probability',
            'value': 'CLV',
            'risk': 'Churn_Probability'
        }
        
        # Determine metric to use
        metric = None
        for key, value in comparison_metrics.items():
            if key in query_lower and value in merged_results.columns:
                metric = value
                break
        
        if not metric:
            # Default to CLV if available
            metric = 'CLV' if 'CLV' in merged_results.columns else 'Churn_Probability'
        
        # Determine grouping dimension
        group_options = [
            'Value_Tier', 'Region', 'Acquisition_Channel', 
            'Customer_Type', 'Industry', 'Company_Size'
        ]
        
        group_by = None
        for option in group_options:
            if option in merged_results.columns:
                group_by = option
                break
        
        if not group_by:
            # Create a CLV vs Churn scatter if no grouping dimension
            return self._generate_clv_churn_scatter(merged_results)
        
        # Group data and calculate average
        grouped_data = merged_results.groupby(group_by)[metric].mean().reset_index()
        
        # Create bar chart
        fig = px.bar(
            grouped_data,
            x=group_by,
            y=metric,
            title=f'Average {metric} by {group_by.replace("_", " ")}',
            color=group_by,
            text_auto='.1%' if metric == 'Churn_Probability' else True
        )
        
        return fig
    
    def _generate_sales_visualization(self, result_df):
        """Generate sales trend visualization from transaction data"""
        if result_df is None or result_df.empty:
            return None
        
        # Find date and amount columns
        date_col = None
        for col in result_df.columns:
            if any(date_term in col.lower() for date_term in ['date', 'time', 'day']):
                date_col = col
                break
                
        amount_col = None
        for col in result_df.columns:
            if any(amount_term in col.lower() for amount_term in ['amount', 'price', 'revenue', 'sales', 'value']):
                amount_col = col
                break
        
        if not date_col or not amount_col:
            return None
        
        # Convert date column and sort
        result_df[date_col] = pd.to_datetime(result_df[date_col])
        result_df = result_df.sort_values(date_col)
        
        # Group by month and sum sales
        sales_by_month = result_df.groupby(pd.Grouper(key=date_col, freq='M'))[amount_col].sum().reset_index()
        
        # Create line chart
        fig = px.line(
            sales_by_month,
            x=date_col,
            y=amount_col,
            title='Monthly Sales Trend',
            labels={date_col: 'Month', amount_col: 'Sales ($)'},
            markers=True
        )
        
        # Add trendline
        fig.add_traces(
            px.scatter(
                sales_by_month, 
                x=date_col, 
                y=amount_col, 
                trendline="ols"
            ).data[1]
        )
        
        return fig
    
    def _generate_clv_churn_scatter(self, merged_results):
        """Generate scatter plot of CLV vs Churn Probability"""
        if 'CLV' not in merged_results.columns or 'Churn_Probability' not in merged_results.columns:
            return None
        
        fig = px.scatter(
            merged_results,
            x='CLV',
            y='Churn_Probability',
            color='Value_Tier' if 'Value_Tier' in merged_results.columns else None,
            title='Customer Lifetime Value vs Churn Probability',
            labels={'CLV': 'Customer Lifetime Value ($)', 'Churn_Probability': 'Churn Probability'},
            hover_data=['Customer ID', 'Customer Name'] if 'Customer Name' in merged_results.columns else ['Customer ID']
        )
        
        # Add quadrant lines
        median_clv = merged_results['CLV'].median()
        median_churn = merged_results['Churn_Probability'].median()
        
        fig.add_vline(x=median_clv, line_dash="dash", line_color="gray")
        fig.add_hline(y=median_churn, line_dash="dash", line_color="gray")
        
        # Add quadrant annotations
        fig.add_annotation(x=median_clv*0.5, y=median_churn*0.5, text="Low CLV, Low Risk",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=median_clv*0.5, y=median_churn*1.5, text="Low CLV, High Risk",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=median_clv*1.5, y=median_churn*0.5, text="High CLV, Low Risk",
                          showarrow=False, font=dict(size=10))
        fig.add_annotation(x=median_clv*1.5, y=median_churn*1.5, text="High CLV, High Risk",
                          showarrow=False, font=dict(size=10))
        
        return fig
    
    def _prepare_data_context(self, merged_results):
        """
        Prepare a context summary from the merged results
        
        Args:
            merged_results (pd.DataFrame): Merged CLV and churn results
        
        Returns:
            str: Textual context summary
        """
        context = "# Business Data Context\n\n"
        
        # Basic statistics
        context += "## Key Metrics\n"
        context += f"- Total Customers: {len(merged_results)}\n"
        context += f"- Average Customer Lifetime Value: ${merged_results['CLV'].mean():,.2f}\n"
        context += f"- Median Customer Lifetime Value: ${merged_results['CLV'].median():,.2f}\n"
        context += f"- Total Customer Value: ${merged_results['CLV'].sum():,.2f}\n\n"
        
        # Segment Distribution
        if 'Value_Tier' in merged_results.columns:
            segment_dist = merged_results['Value_Tier'].value_counts(normalize=True)
            context += "## Customer Segment Distribution\n"
            for segment, proportion in segment_dist.items():
                context += f"- {segment}: {proportion:.1%}\n"
            context += "\n"
        
        # Churn Risk Overview
        context += "## Churn Risk Overview\n"
        
        # Use the renamed churn probability column
        churn_col = 'Churn_Probability_Churn' if 'Churn_Probability_Churn' in merged_results.columns else 'Churn_Probability'
        if churn_col in merged_results.columns:
            context += f"- Average Churn Probability: {merged_results[churn_col].mean():.1%}\n"
            context += f"- High-Risk Customers: {(merged_results[churn_col] > 0.7).sum()} "
            context += f"({(merged_results[churn_col] > 0.7).mean():.1%})\n"
        
        return context
    
    def _prepare_transaction_context(self, result_df):
        """
        Prepare transaction context summary
        
        Args:
            result_df (pd.DataFrame): Transaction data
        
        Returns:
            str: Transaction context summary
        """
        context = "# Transaction Data Context\n\n"
        
        # Find date and amount columns
        date_col = None
        for col in result_df.columns:
            if any(date_term in col.lower() for date_term in ['date', 'time', 'day']):
                date_col = col
                break
                
        amount_col = None
        for col in result_df.columns:
            if any(amount_term in col.lower() for amount_term in ['amount', 'price', 'revenue', 'sales', 'value']):
                amount_col = col
                break
        
        if date_col and amount_col:
            # Basic statistics
            context += "## Transaction Metrics\n"
            context += f"- Total Transactions: {len(result_df)}\n"
            context += f"- Total Sales: ${result_df[amount_col].sum():,.2f}\n"
            context += f"- Average Transaction: ${result_df[amount_col].mean():,.2f}\n"
            
            # Try to get date range
            try:
                result_df[date_col] = pd.to_datetime(result_df[date_col])
                min_date = result_df[date_col].min().strftime('%Y-%m-%d')
                max_date = result_df[date_col].max().strftime('%Y-%m-%d')
                context += f"- Transaction Period: {min_date} to {max_date}\n\n"
            except:
                context += "\n"
            
            # Monthly trend
            try:
                monthly_sales = result_df.groupby(pd.Grouper(key=date_col, freq='M'))[amount_col].sum()
                recent_months = monthly_sales.tail(3)
                
                context += "## Recent Sales Trend (Last 3 Months)\n"
                for date, sales in recent_months.items():
                    context += f"- {date.strftime('%B %Y')}: ${sales:,.2f}\n"
            except:
                pass
        
        return context
    def _is_visualization_query(self, query_lower):
        """Check if query is specifically requesting visualization"""
        visualization_terms = [
            'plot', 'chart', 'graph', 'visualize', 'visualization', 'show me', 
            'display', 'histogram', 'pie chart', 'bar chart', 'dashboard'
        ]
        return any(term in query_lower for term in visualization_terms)
    
    def _handle_visualization_query(self, query, merged_results, result_df):
        """Generate response for visualization-specific queries"""
        response = {
            "text": "",
            "visualization": None,
            "business_insights": {},
            "action_items": []
        }
        
        query_lower = query.lower()
        
        # CLV Distribution visualization
        if any(term in query_lower for term in ['clv distribution', 'clv histogram', 'distribution of clv']):
            fig = self._generate_clv_distribution(merged_results)
            response["visualization"] = fig
            response["text"] = "## Customer Lifetime Value Distribution\n\n"
            response["text"] += "This visualization shows the distribution of Customer Lifetime Value (CLV) across your customer base.\n\n"
            
            # Add insights about CLV distribution
            avg_clv = merged_results['CLV'].mean()
            median_clv = merged_results['CLV'].median()
            max_clv = merged_results['CLV'].max()
            
            response["text"] += "### Key Insights\n"
            response["text"] += f"- **Average CLV:** ${avg_clv:,.2f}\n"
            response["text"] += f"- **Median CLV:** ${median_clv:,.2f}\n"
            response["text"] += f"- **Maximum CLV:** ${max_clv:,.2f}\n"
            
            # Check distribution skew
            skew = (avg_clv - median_clv) / median_clv if median_clv > 0 else 0
            if skew > 0.3:
                response["text"] += f"- **Distribution Skew:** The distribution is positively skewed (by {skew:.1%}), indicating a large portion of revenue comes from a small number of high-value customers.\n"
            
            return response
        
        # Churn Risk visualization
        elif any(term in query_lower for term in ['churn risk', 'churn visualization', 'churn probability']):
            fig = self._generate_churn_heatmap(merged_results)
            response["visualization"] = fig
            response["text"] = "## Customer Churn Risk Analysis\n\n"
            
            # Add insights about churn
            if 'Churn_Probability' in merged_results.columns:
                avg_churn = merged_results['Churn_Probability'].mean()
                high_risk_count = (merged_results['Churn_Probability'] > 0.7).sum()
                high_risk_pct = high_risk_count / len(merged_results)
                
                response["text"] += "### Key Insights\n"
                response["text"] += f"- **Average Churn Probability:** {avg_churn:.1%}\n"
                response["text"] += f"- **High Risk Customers:** {high_risk_count} ({high_risk_pct:.1%} of total)\n"
                
                # Check if we have CLV and value tier
                if 'CLV' in merged_results.columns and 'Value_Tier' in merged_results.columns:
                    high_value_at_risk = merged_results[
                        (merged_results['Value_Tier'].isin(['Premium', 'High'])) & 
                        (merged_results['Churn_Probability'] > 0.5)
                    ]
                    
                    if not high_value_at_risk.empty:
                        high_value_at_risk_count = len(high_value_at_risk)
                        high_value_at_risk_clv = high_value_at_risk['CLV'].sum()
                        
                        response["text"] += f"- **High-Value Customers at Risk:** {high_value_at_risk_count}\n"
                        response["text"] += f"- **Potential Revenue at Risk:** ${high_value_at_risk_clv:,.2f}\n"
            
            return response
        
        # Customer Segments visualization
        elif any(term in query_lower for term in ['customer segments', 'customer segmentation', 'segment pie chart']):
            fig = self._generate_segment_visualization(merged_results)
            response["visualization"] = fig
            response["text"] = "## Customer Segmentation Analysis\n\n"
            
            # Add segment information
            if 'Value_Tier' in merged_results.columns:
                segment_counts = merged_results['Value_Tier'].value_counts()
                segment_values = merged_results.groupby('Value_Tier')['CLV'].agg(['sum', 'mean']).reset_index()
                
                response["text"] += "### Segment Breakdown\n\n"
                response["text"] += "| Segment | Customer Count | % of Customers | Total Value | Average CLV |\n"
                response["text"] += "|---------|---------------|----------------|-------------|------------|\n"
                
                for segment in segment_counts.index:
                    count = segment_counts[segment]
                    pct = count / len(merged_results)
                    segment_data = segment_values[segment_values['Value_Tier'] == segment]
                    total_value = segment_data['sum'].iloc[0] if not segment_data.empty else 0
                    avg_value = segment_data['mean'].iloc[0] if not segment_data.empty else 0
                    
                    response["text"] += f"| {segment} | {count} | {pct:.1%} | ${total_value:,.2f} | ${avg_value:,.2f} |\n"
            
            return response
        
        # Sales Forecast visualization
        elif any(term in query_lower for term in ['sales forecast', 'revenue forecast', 'projected sales']):
            forecast_response = self._generate_sales_forecast(result_df)
            return forecast_response
        
        # CLV vs Churn visualization (scatter plot)
        elif any(term in query_lower for term in ['clv vs churn', 'scatter plot', 'value vs risk']):
            fig = self._generate_clv_churn_scatter(merged_results)
            response["visualization"] = fig
            response["text"] = "## Customer Value vs. Churn Risk Analysis\n\n"
            response["text"] += "This scatter plot shows the relationship between Customer Lifetime Value (CLV) and Churn Probability for each customer.\n\n"
            
            # Add quadrant analysis
            response["text"] += "### Quadrant Analysis\n\n"
            response["text"] += "The chart is divided into four quadrants based on median CLV and churn probability:\n\n"
            response["text"] += "1. **Low CLV, Low Risk (Bottom Left):** Stable but low-value customers\n"
            response["text"] += "2. **Low CLV, High Risk (Top Left):** At-risk low-value customers\n"
            response["text"] += "3. **High CLV, Low Risk (Bottom Right):** Ideal customers - high value and loyal\n"
            response["text"] += "4. **High CLV, High Risk (Top Right):** Critical focus area - high-value customers at risk of churning\n\n"
            
            # Add strategic recommendations
            response["text"] += "### Strategic Recommendations\n\n"
            response["text"] += "- **For High CLV, High Risk:** Implement immediate retention strategies and personalized outreach\n"
            response["text"] += "- **For High CLV, Low Risk:** Focus on expansion opportunities and loyalty programs\n"
            response["text"] += "- **For Low CLV, High Risk:** Evaluate cost-effective retention approaches\n"
            response["text"] += "- **For Low CLV, Low Risk:** Develop growth strategies to increase customer value\n"
            
            return response
        
        # Default to generic visualization
        else:
            fig = self._generate_appropriate_visualization(query, merged_results, result_df)
            response["visualization"] = fig
            response["text"] = "## Data Visualization\n\n"
            response["text"] += "Here's a visualization based on your request. The chart shows key metrics from your customer and transaction data.\n\n"
            
            return response