import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.database.store import get_alerts


def main():
    st.title("Alert History")

    if not st.session_state.get("authenticated", False):
        st.warning("Please log in to access this page.")
        return

    company_id = st.session_state.company_id

    col1, col2 = st.columns(2)

    with col1:
        alert_type_label = st.selectbox("Alert Type", options=["All", "Churn Risk", "Sales Risk"])

    with col2:
        time_period = st.selectbox("Time Period", options=["Last 7 days", "Last 30 days", "Last 90 days", "All time"])

    alert_type = None
    if alert_type_label != "All":
        alert_type = 'churn' if alert_type_label == "Churn Risk" else 'sales'

    today = datetime.now()
    cutoff_map = {
        "Last 7 days": today - timedelta(days=7),
        "Last 30 days": today - timedelta(days=30),
        "Last 90 days": today - timedelta(days=90),
        "All time": datetime(2000, 1, 1),
    }
    since = cutoff_map[time_period]

    alerts = get_alerts(company_id, alert_type=alert_type, since=since)

    if not alerts:
        st.info("No alerts found matching your criteria.")
        return

    alert_data = []
    for alert in alerts:
        details = alert['details']
        if alert['type'] == 'churn':
            description = f"{details.get('customer_count', 0)} customers at risk, ${details.get('total_value', 0):,.2f} value"
        else:
            description = f"{details.get('decline_percentage', 0):.1%} decline, ${details.get('change_amount', 0):,.2f} lost"

        alert_data.append({
            'id': alert['id'],
            'Date': datetime.fromisoformat(alert['created_at']).strftime('%Y-%m-%d %H:%M'),
            'Type': 'Churn Risk' if alert['type'] == 'churn' else 'Sales Decline',
            'Risk Level': f"{alert['risk_level']:.1%}",
            'Description': description,
        })

    df = pd.DataFrame(alert_data)
    st.dataframe(df.drop(columns=['id']), use_container_width=True)

    st.subheader("Alert Details")
    selected_label = st.selectbox(
        "Select an alert to view details",
        options=[f"{row['Date']} - {row['Type']}" for _, row in df.iterrows()]
    )

    selected_row = df[df['Date'] + ' - ' + df['Type'] == selected_label].iloc[0]
    selected_alert = next(a for a in alerts if a['id'] == selected_row['id'])
    details = selected_alert['details']

    with st.expander("Alert Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Alert Type:**", "Churn Risk" if selected_alert['type'] == 'churn' else "Sales Decline")
            st.write("**Created At:**", datetime.fromisoformat(selected_alert['created_at']).strftime('%Y-%m-%d %H:%M:%S'))
            st.write("**Risk Level:**", f"{selected_alert['risk_level']:.1%}")
        with col2:
            st.write("**Delivery:**", "In-app notification")

        st.subheader("Details")

        if selected_alert['type'] == 'churn':
            st.write(f"**Customers at Risk:** {details.get('customer_count', 0)}")
            st.write(f"**Total Value at Risk:** ${details.get('total_value', 0):,.2f}")
            st.write("**Highest Risk Customers:**")
            for customer in details.get('highest_risk_customers', []):
                st.write(f"- {customer.get('name', 'Unknown')}: {customer.get('churn_risk', 0):.1%} risk, ${customer.get('clv', 0):,.2f} value")
        else:
            st.write(f"**Current Period Sales:** ${details.get('current_sales', 0):,.2f}")
            st.write(f"**Previous Period Sales:** ${details.get('previous_sales', 0):,.2f}")
            st.write(f"**Change Amount:** -${details.get('change_amount', 0):,.2f}")
            st.write(f"**Decline Percentage:** {details.get('decline_percentage', 0):.1%}")
            st.write("**Contributing Factors:**")
            for factor in details.get('factors', []):
                st.write(f"- {factor}")


if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None

    main()
