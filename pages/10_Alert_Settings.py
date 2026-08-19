import streamlit as st

from src.database.store import get_alert_settings, save_alert_settings
from src.alerts.risk_alerts import CHURN_DEFAULTS, SALES_DEFAULTS


def main():
    st.title("Alert Settings")
    st.caption("Alerts are generated automatically while analyzing your data and shown in-app under Alert History and the sidebar.")

    if not st.session_state.get("authenticated", False):
        st.warning("Please log in to access this page.")
        return

    company_id = st.session_state.company_id

    tab1, tab2 = st.tabs(["Churn Risk Alerts", "Sales Risk Alerts"])

    with tab1:
        st.subheader("Churn Risk Thresholds")
        settings = get_alert_settings(company_id, 'churn', CHURN_DEFAULTS)

        with st.form("churn_alert_settings"):
            threshold = st.slider(
                "Churn probability threshold",
                min_value=0.1, max_value=0.95,
                value=float(settings.get('threshold') or CHURN_DEFAULTS['threshold']),
                step=0.05,
                help="Customers above this churn probability count as high risk."
            )
            min_customers = st.number_input(
                "Minimum at-risk customers to trigger an alert",
                min_value=1, max_value=1000,
                value=int(settings.get('min_customers') or CHURN_DEFAULTS['min_customers'])
            )
            frequency = st.selectbox(
                "Alert frequency (minimum time between repeat alerts)",
                options=['daily', 'weekly'],
                index=['daily', 'weekly'].index(settings.get('alert_frequency') or CHURN_DEFAULTS['alert_frequency'])
            )
            submitted = st.form_submit_button("Save Churn Alert Settings", type="primary")

        if submitted:
            save_alert_settings(company_id, 'churn', threshold=threshold,
                                 min_customers=min_customers, alert_frequency=frequency)
            st.success("Churn alert settings saved.")

    with tab2:
        st.subheader("Sales Decline Thresholds")
        settings = get_alert_settings(company_id, 'sales', SALES_DEFAULTS)

        with st.form("sales_alert_settings"):
            threshold = st.slider(
                "Sales decline threshold (month-over-month)",
                min_value=0.05, max_value=0.9,
                value=float(settings.get('threshold') or SALES_DEFAULTS['threshold']),
                step=0.05,
                help="A month-over-month sales drop above this percentage triggers an alert."
            )
            comparison_periods = st.number_input(
                "Comparison periods (months of history required)",
                min_value=2, max_value=24,
                value=int(settings.get('comparison_periods') or SALES_DEFAULTS['comparison_periods'])
            )
            frequency = st.selectbox(
                "Alert frequency (minimum time between repeat alerts)",
                options=['daily', 'weekly'],
                index=['daily', 'weekly'].index(settings.get('alert_frequency') or SALES_DEFAULTS['alert_frequency']),
                key="sales_frequency"
            )
            submitted = st.form_submit_button("Save Sales Alert Settings", type="primary")

        if submitted:
            save_alert_settings(company_id, 'sales', threshold=threshold,
                                 comparison_periods=comparison_periods, alert_frequency=frequency)
            st.success("Sales alert settings saved.")


if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None

    main()
