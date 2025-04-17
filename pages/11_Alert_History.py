import streamlit as st
import pandas as pd
from firebase_admin import firestore
from datetime import datetime, timedelta

def main():
    st.title("Alert History")
    
    if not st.session_state.authenticated:
        st.warning("Please log in to access this page.")
        return
    
    company_id = st.session_state.company_id
    
    # Get Firestore instance
    db = firestore.client()
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_type = st.selectbox(
            "Alert Type",
            options=["All", "Churn Risk", "Sales Risk"]
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    
    with col3:
        status = st.selectbox(
            "Status",
            options=["All", "Sent", "Pending"]
        )
    
    # Build query
    alerts_ref = db.collection('companies').document(company_id).collection('alerts')
    query = alerts_ref.order_by('created_at', direction='DESCENDING')
    
    # Apply filters
    if alert_type != "All":
        query = query.where('type', '==', 'churn' if alert_type == "Churn Risk" else 'sales')
    
    if status != "All":
        query = query.where('status', '==', status.lower())
    
    # Execute query
    alerts = query.get()
    
    # Filter by time period
    today = datetime.now()
    
    if time_period == "Last 7 days":
        cutoff_date = today - timedelta(days=7)
    elif time_period == "Last 30 days":
        cutoff_date = today - timedelta(days=30)
    elif time_period == "Last 90 days":
        cutoff_date = today - timedelta(days=90)
    else:  # All time
        cutoff_date = datetime(2000, 1, 1)  # Very old date
    
    # Process alerts
    alert_data = []
    
    for alert in alerts:
        alert_dict = alert.to_dict()
        alert_date = alert_dict.get('created_at')
        
        if alert_date and alert_date.replace(tzinfo=None) >= cutoff_date:
            # Format data for display
            if alert_dict.get('type') == 'churn':
                details = alert_dict.get('details', {})
                description = f"{details.get('customer_count', 0)} customers at risk, ${details.get('total_value', 0):,.2f} value"
            else:  # sales
                details = alert_dict.get('details', {})
                description = f"{details.get('decline_percentage', 0):.1%} decline, ${details.get('change_amount', 0):,.2f} lost"
            
            alert_data.append({
                'id': alert.id,
                'Date': alert_date.strftime('%Y-%m-%d %H:%M') if alert_date else 'N/A',
                'Type': 'Churn Risk' if alert_dict.get('type') == 'churn' else 'Sales Decline',
                'Risk Level': f"{alert_dict.get('risk_level', 0):.1%}",
                'Description': description,
                'Status': 'Sent' if alert_dict.get('status') == 'sent' else 'Pending',
                'Recipients': ", ".join(alert_dict.get('recipients', []))[:50] + ("..." if len(", ".join(alert_dict.get('recipients', []))) > 50 else "")
            })
    
    # Display alerts
    if alert_data:
        df = pd.DataFrame(alert_data)
        df_display = df.drop(columns=['id'])  # Don't show ID in the table
        
        st.dataframe(df_display, use_container_width=True)
        
        # Alert details section
        st.subheader("Alert Details")
        
        selected_alert_id = st.selectbox(
            "Select an alert to view details",
            options=[f"{row['Date']} - {row['Type']}" for _, row in df.iterrows()],
            format_func=lambda x: x
        )
        
        if selected_alert_id:
            # Find the selected alert
            selected_index = df.index[df['Date'] + ' - ' + df['Type'] == selected_alert_id][0]
            selected_alert_id = df.iloc[selected_index]['id']
            
            # Get alert details
            alert_doc = alerts_ref.document(selected_alert_id).get()
            
            if alert_doc.exists:
                alert_data = alert_doc.to_dict()
                
                with st.expander("Alert Details", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Alert Type:**", "Churn Risk" if alert_data.get('type') == 'churn' else "Sales Decline")
                        st.write("**Created At:**", alert_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if alert_data.get('created_at') else 'N/A')
                        st.write("**Risk Level:**", f"{alert_data.get('risk_level', 0):.1%}")
                        st.write("**Status:**", "Sent" if alert_data.get('status') == 'sent' else "Pending")
                    
                    with col2:
                        if alert_data.get('status') == 'sent':
                            st.write("**Sent At:**", alert_data.get('sent_at').strftime('%Y-%m-%d %H:%M:%S') if alert_data.get('sent_at') else 'N/A')
                        
                        st.write("**Recipients:**")
                        for recipient in alert_data.get('recipients', []):
                            st.write(f"- {recipient}")
                    
                    st.subheader("Alert Details")
                    
                    details = alert_data.get('details', {})
                    
                    if alert_data.get('type') == 'churn':
                        st.write(f"**Customers at Risk:** {details.get('customer_count', 0)}")
                        st.write(f"**Total Value at Risk:** ${details.get('total_value', 0):,.2f}")
                        
                        st.write("**Highest Risk Customers:**")
                        for customer in details.get('highest_risk_customers', []):
                            st.write(f"- {customer.get('name', 'Unknown')}: {customer.get('churn_risk', 0):.1%} risk, ${customer.get('clv', 0):,.2f} value")
                    
                    else:  # sales
                        st.write(f"**Current Period Sales:** ${details.get('current_sales', 0):,.2f}")
                        st.write(f"**Previous Period Sales:** ${details.get('previous_sales', 0):,.2f}")
                        st.write(f"**Change Amount:** -${details.get('change_amount', 0):,.2f}")
                        st.write(f"**Decline Percentage:** {details.get('decline_percentage', 0):.1%}")
                        
                        st.write("**Contributing Factors:**")
                        for factor in details.get('factors', []):
                            st.write(f"- {factor}")
                    
                    # Resend option
                    if st.button("Resend Alert"):
                        from src.utils.email_service import send_email
                        from src.alerts.risk_alerts import generate_churn_alert_html, generate_sales_alert_html
                        
                        # Get company information
                        company_doc = db.collection('companies').document(company_id).get()
                        company_name = company_doc.to_dict().get('name', 'Your Company')
                        
                        # Generate email content
                        if alert_data.get('type') == 'churn':
                            html_content = generate_churn_alert_html(company_name, details)
                            subject = f"⚠️ BizNexus AI Alert: High Churn Risk Detected for {company_name}"
                        else:  # sales
                            html_content = generate_sales_alert_html(company_name, details)
                            subject = f"⚠️ BizNexus AI Alert: Sales Decline Detected for {company_name}"
                        
                        # Send email
                        user_id = st.session_state.user_info.get('user_id') if st.session_state.user_info else None
                        email_sent = send_email(alert_data.get('recipients', []), subject, html_content, user_id)
                        
                        if email_sent:
                            # Update alert status
                            alerts_ref.document(selected_alert_id).update({
                                'status': 'sent',
                                'sent_at': firestore.SERVER_TIMESTAMP
                            })
                            
                            st.success("Alert resent successfully!")
                        else:
                            st.error("Failed to resend alert. Please check email settings.")
    else:
        st.info("No alerts found matching your criteria.")

if __name__ == "__main__":
    # Initialize session state if not already done
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None
        
    main()