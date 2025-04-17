from firebase_admin import firestore
from datetime import datetime
import pandas as pd
from src.utils.email_service import send_email
import logging
def generate_churn_alert_html(company_name, risk_data):
    """Generate HTML content for churn risk alert"""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4da6ff; color: white; padding: 10px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f9f9f9; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
            .risk-high {{ color: #d9534f; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>BizNexus AI - High Churn Risk Alert</h2>
            </div>
            <div class="content">
                <p>Dear {company_name} Analyst,</p>
                
                <p>Our system has detected <span class="risk-high">{risk_data['customer_count']} customers</span> with a high risk of churn, 
                representing approximately <span class="risk-high">${risk_data['total_value']:,.2f}</span> in potential lost revenue.</p>
                
                <h3>Top High-Risk Customers:</h3>
                <table>
                    <tr>
                        <th>Customer</th>
                        <th>Churn Risk</th>
                        <th>Lifetime Value</th>
                    </tr>
    """
    
    # Add top 5 customers to the table
    for customer in risk_data['highest_risk_customers'][:5]:
        html += f"""
                    <tr>
                        <td>{customer.get('name', customer.get('id', 'Unknown'))}</td>
                        <td class="risk-high">{customer.get('churn_risk', 0):.1%}</td>
                        <td>${customer.get('clv', 0):,.2f}</td>
                    </tr>
        """
    
    html += f"""
                </table>
                
                <p>Please log in to your BizNexus AI dashboard to view detailed churn analysis and recommended actions.</p>
                
                <p>Best regards,<br>
                BizNexus AI Team</p>
            </div>
            <div class="footer">
                <p>This is an automated alert from BizNexus AI. Please do not reply to this email.</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def generate_sales_alert_html(company_name, risk_data):
    """Generate HTML content for sales risk alert"""
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4da6ff; color: white; padding: 10px; text-align: center; }}
            .content {{ padding: 20px; background-color: #f9f9f9; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
            .risk-high {{ color: #d9534f; font-weight: bold; }}
            .trend-down {{ color: #d9534f; }}
            .trend-up {{ color: #5cb85c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>BizNexus AI - Sales Risk Alert</h2>
            </div>
            <div class="content">
                <p>Dear {company_name} Analyst,</p>
                
                <p>Our system has detected a <span class="risk-high">{risk_data['decline_percentage']:.1%} decline</span> in sales 
                compared to the previous period.</p>
                
                <h3>Sales Performance Summary:</h3>
                <ul>
                    <li>Current Period: <span class="trend-down">${risk_data['current_sales']:,.2f}</span></li>
                    <li>Previous Period: <span>${risk_data['previous_sales']:,.2f}</span></li>
                    <li>Change: <span class="trend-down">-${risk_data['change_amount']:,.2f}</span></li>
                </ul>
                
                <p>Primary factors contributing to this decline:</p>
                <ul>
    """
    
    # Add factors to the list
    for factor in risk_data['factors']:
        html += f"<li>{factor}</li>"
    
    html += f"""
                </ul>
                
                <p>Please log in to your BizNexus AI dashboard to view detailed sales analysis and recommended actions.</p>
                
                <p>Best regards,<br>
                BizNexus AI Team</p>
            </div>
            <div class="footer">
                <p>This is an automated alert from BizNexus AI. Please do not reply to this email.</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def check_churn_risk(company_id, merged_results):
    """
    Check for high churn risk and trigger alerts if needed
    
    Args:
        company_id (str): Company ID
        merged_results (pd.DataFrame): Merged CLV and churn data
    
    Returns:
        bool: True if alert was generated, False otherwise
    """
    db = firestore.client()
    
    # Get alert settings
    settings_ref = db.collection('companies').document(company_id).collection('alert_settings').document('churn')
    settings = settings_ref.get()
    
    if not settings.exists:
        # Create default settings if not exists
        settings_ref.set({
            'threshold': 0.7,
            'min_customers': 3,
            'alert_frequency': 'daily',
            'recipients': []
        })
        settings = settings_ref.get()
    
    settings_data = settings.to_dict()
    threshold = settings_data.get('threshold', 0.7)
    min_customers = settings_data.get('min_customers', 3)
    
    # Identify high risk customers
    churn_col = 'Churn_Probability' if 'Churn_Probability' in merged_results.columns else 'Churn_Probability_Churn'
    if churn_col not in merged_results.columns:
        return False
    
    high_risk = merged_results[merged_results[churn_col] > threshold]
    
    if len(high_risk) >= min_customers:
        # Check if we should send an alert based on frequency
        alert_ref = db.collection('companies').document(company_id).collection('alerts')
        last_alert = alert_ref.where('type', '==', 'churn').order_by('created_at', direction='DESCENDING').limit(1).get()
        
        should_alert = True
        if last_alert:
            last_alert_time = last_alert[0].to_dict().get('created_at')
            frequency = settings_data.get('alert_frequency', 'daily')
            
            if frequency == 'daily':
                # Check if last alert was more than 24 hours ago
                should_alert = (datetime.now() - last_alert_time).total_seconds() > 86400
            elif frequency == 'weekly':
                # Check if last alert was more than 7 days ago
                should_alert = (datetime.now() - last_alert_time).total_seconds() > 604800
        
        if should_alert:
            # Prepare alert data
            total_value = high_risk['CLV'].sum() if 'CLV' in high_risk.columns else 0
            
            # Get top high-risk customers
            top_risk = high_risk.nlargest(10, 'CLV' if 'CLV' in high_risk.columns else churn_col)
            highest_risk_customers = []
            
            for _, row in top_risk.iterrows():
                customer = {
                    'id': row.get('Customer ID', ''),
                    'name': row.get('Customer Name', f"Customer {row.get('Customer ID', '')}"),
                    'churn_risk': row.get(churn_col, 0),
                    'clv': row.get('CLV', 0)
                }
                highest_risk_customers.append(customer)
            
            # Create alert data
            alert_data = {
                'created_at': firestore.SERVER_TIMESTAMP,
                'type': 'churn',
                'risk_level': high_risk[churn_col].mean(),
                'details': {
                    'customer_count': len(high_risk),
                    'total_value': total_value,
                    'highest_risk_customers': highest_risk_customers
                },
                'status': 'pending',
                'recipients': settings_data.get('recipients', [])
            }
            
            # Save alert
            new_alert_ref = alert_ref.document()
            new_alert_ref.set(alert_data)
            
            # Get company information
            company_doc = db.collection('companies').document(company_id).get()
            company_name = company_doc.to_dict().get('name', 'Your Company')
            
            # Get company analysts
            users_ref = db.collection('users').where('company_id', '==', company_id).get()
            recipients = settings_data.get('recipients', [])
            
            for user in users_ref:
                user_data = user.to_dict()
                if user_data.get('email') and user_data.get('email') not in recipients:
                    recipients.append(user_data.get('email'))
            
            if recipients:
                # Generate email content
                html_content = generate_churn_alert_html(company_name, alert_data['details'])
                
                # Send email
                subject = f"⚠️ BizNexus AI Alert: High Churn Risk Detected for {company_name}"
                email_sent = send_email(recipients, subject, html_content)
                
                # Update alert status
                if email_sent:
                    new_alert_ref.update({
                        'status': 'sent',
                        'sent_at': firestore.SERVER_TIMESTAMP,
                        'recipients': recipients
                    })
                
                return email_sent
    
    return False



def check_sales_risk(company_id, sales_data):
    """
    Check for sales risk and trigger alerts if needed with improved error handling
    
    Args:
        company_id (str): Company ID
        sales_data (pd.DataFrame): Transaction data
    
    Returns:
        bool: True if alert was generated, False otherwise
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        db = firestore.client()
        
        # Get alert settings with improved error handling
        settings_ref = db.collection('companies').document(company_id).collection('alert_settings').document('sales')
        
        try:
            settings = settings_ref.get()
        except Exception as settings_fetch_error:
            logger.warning(f"Could not fetch sales alert settings: {settings_fetch_error}")
            # Create default settings if not exists
            settings_ref.set({
                'threshold': 0.15,  # 15% decline
                'comparison_periods': 2,
                'alert_frequency': 'weekly',
                'recipients': []
            })
            settings = settings_ref.get()
        
        # Fallback settings if document doesn't exist
        if not settings.exists:
            settings_data = {
                'threshold': 0.15,
                'comparison_periods': 2,
                'alert_frequency': 'weekly',
                'recipients': []
            }
        else:
            settings_data = settings.to_dict()
        
        threshold = settings_data.get('threshold', 0.15)
        comparison_periods = settings_data.get('comparison_periods', 2)
        
        # Find date and amount columns
        date_col = None
        for col in sales_data.columns:
            if any(term in col.lower() for term in ['date', 'time', 'day']):
                date_col = col
                break
        
        amount_col = None
        for col in sales_data.columns:
            if any(term in col.lower() for term in ['amount', 'price', 'revenue', 'sales']):
                amount_col = col
                break
        
        if not date_col or not amount_col:
            logger.warning("Could not find date or amount column")
            return False
        
        # Convert date column to datetime
        sales_data[date_col] = pd.to_datetime(sales_data[date_col])
        
        # Group by month and calculate sales
        monthly_sales = sales_data.groupby(pd.Grouper(key=date_col, freq='M'))[amount_col].sum().reset_index()
        monthly_sales = monthly_sales.sort_values(date_col, ascending=False)
        
        if len(monthly_sales) < comparison_periods + 1:
            logger.info("Not enough data for comparison")
            return False
        
        # Calculate current and previous periods
        current_sales = monthly_sales.iloc[0][amount_col]
        previous_sales = monthly_sales.iloc[1][amount_col]
        
        # Calculate decline percentage
        if previous_sales > 0:
            decline_percentage = (previous_sales - current_sales) / previous_sales
        else:
            decline_percentage = 0
        
        if decline_percentage > threshold:
            # Check if we should send an alert based on frequency
            alerts_ref = db.collection('companies').document(company_id).collection('alerts')
            
            try:
                last_alert = alerts_ref.where('type', '==', 'sales').order_by('created_at', direction='DESCENDING').limit(1).get()
            except Exception as alert_query_error:
                logger.warning(f"Could not query previous alerts: {alert_query_error}")
                last_alert = []
            
            should_alert = True
            if last_alert:
                last_alert_time = last_alert[0].to_dict().get('created_at')
                frequency = settings_data.get('alert_frequency', 'weekly')
                
                if frequency == 'daily':
                    # Check if last alert was more than 24 hours ago
                    should_alert = (datetime.now() - last_alert_time).total_seconds() > 86400
                elif frequency == 'weekly':
                    # Check if last alert was more than 7 days ago
                    should_alert = (datetime.now() - last_alert_time).total_seconds() > 604800
            
            if should_alert:
                # Analyze factors for decline
                factors = []
                
                # Check customer count
                current_month = monthly_sales.iloc[0][date_col].to_pydatetime()
                prev_month = monthly_sales.iloc[1][date_col].to_pydatetime()
                
                try:
                    current_customers = len(sales_data[sales_data[date_col].dt.month == current_month.month]['Customer ID'].unique())
                    prev_customers = len(sales_data[sales_data[date_col].dt.month == prev_month.month]['Customer ID'].unique())
                    
                    if current_customers < prev_customers:
                        customer_decline = (prev_customers - current_customers) / prev_customers
                        factors.append(f"Customer count decreased by {customer_decline:.1%} ({prev_customers} to {current_customers})")
                except Exception as customer_count_error:
                    logger.warning(f"Could not calculate customer count: {customer_count_error}")
                
                # Check average order value
                try:
                    current_aov = current_sales / max(1, current_customers)
                    prev_aov = previous_sales / max(1, prev_customers)
                    
                    if current_aov < prev_aov:
                        aov_decline = (prev_aov - current_aov) / prev_aov
                        factors.append(f"Average order value decreased by {aov_decline:.1%} (${prev_aov:.2f} to ${current_aov:.2f})")
                except Exception as aov_error:
                    logger.warning(f"Could not calculate average order value: {aov_error}")
                
                # Add default factor if none found
                if not factors:
                    factors.append("General sales decline across customer base")
                
                # Create alert data
                alert_data = {
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'type': 'sales',
                    'risk_level': decline_percentage,
                    'details': {
                        'current_sales': current_sales,
                        'previous_sales': previous_sales,
                        'change_amount': previous_sales - current_sales,
                        'decline_percentage': decline_percentage,
                        'current_month': current_month.strftime('%B %Y'),
                        'previous_month': prev_month.strftime('%B %Y'),
                        'factors': factors
                    },
                    'status': 'pending',
                    'recipients': settings_data.get('recipients', [])
                }
                
                # Save alert
                try:
                    new_alert_ref = alerts_ref.document()
                    new_alert_ref.set(alert_data)
                except Exception as alert_save_error:
                    logger.error(f"Could not save sales alert: {alert_save_error}")
                    return False
                
                # Get company information
                try:
                    company_doc = db.collection('companies').document(company_id).get()
                    company_name = company_doc.to_dict().get('name', 'Your Company')
                except Exception as company_info_error:
                    logger.warning(f"Could not retrieve company information: {company_info_error}")
                    company_name = 'Your Company'
                
                # Get company analysts
                try:
                    users_ref = db.collection('users').where('company_id', '==', company_id).get()
                    recipients = settings_data.get('recipients', [])
                    
                    for user in users_ref:
                        user_data = user.to_dict()
                        if user_data.get('email') and user_data.get('email') not in recipients:
                            recipients.append(user_data.get('email'))
                except Exception as user_fetch_error:
                    logger.warning(f"Could not fetch company users: {user_fetch_error}")
                
                if recipients:
                    # Generate email content
                    try:
                        from src.utils.email_service import send_email
                        from src.alerts.risk_alerts import generate_sales_alert_html
                        
                        html_content = generate_sales_alert_html(company_name, alert_data['details'])
                        
                        # Send email
                        subject = f"⚠️ BizNexus AI Alert: Sales Decline Detected for {company_name}"
                        email_sent = send_email(recipients, subject, html_content)
                        
                        # Update alert status
                        if email_sent:
                            try:
                                new_alert_ref.update({
                                    'status': 'sent',
                                    'sent_at': firestore.SERVER_TIMESTAMP,
                                    'recipients': recipients
                                })
                            except Exception as alert_update_error:
                                logger.warning(f"Could not update alert status: {alert_update_error}")
                            
                            return True
                    except Exception as email_error:
                        logger.error(f"Email sending failed: {email_error}")
        
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error in sales risk check: {e}")
        return False