from datetime import datetime
import logging

import pandas as pd

from src.database.store import get_alert_settings, get_last_alert, save_alert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHURN_DEFAULTS = {'threshold': 0.7, 'min_customers': 3, 'alert_frequency': 'daily'}
SALES_DEFAULTS = {'threshold': 0.15, 'comparison_periods': 2, 'alert_frequency': 'weekly'}


def _should_alert(last_alert, frequency):
    if not last_alert:
        return True
    last_alert_time = datetime.fromisoformat(last_alert['created_at'])
    elapsed = (datetime.now() - last_alert_time).total_seconds()
    if frequency == 'daily':
        return elapsed > 86400
    if frequency == 'weekly':
        return elapsed > 604800
    return True


def check_churn_risk(company_id, merged_results):
    """
    Check for high churn risk and record an in-app alert if the threshold is met.

    Args:
        company_id (str): Company ID
        merged_results (pd.DataFrame): Merged CLV and churn data

    Returns:
        bool: True if an alert was recorded, False otherwise
    """
    settings = get_alert_settings(company_id, 'churn', CHURN_DEFAULTS)
    threshold = settings.get('threshold') or CHURN_DEFAULTS['threshold']
    min_customers = settings.get('min_customers') or CHURN_DEFAULTS['min_customers']
    frequency = settings.get('alert_frequency') or CHURN_DEFAULTS['alert_frequency']

    # Prefer the suffixed column: when CLV results (which already carry a
    # crude bootstrap 'Churn_Probability' stand-in from feature engineering)
    # are merged with real churn model predictions using
    # suffixes=('', '_Churn'), the real model's output becomes
    # 'Churn_Probability_Churn' while the unsuffixed name keeps the cruder
    # fallback - see src/agent/tools.py's _churn_col for the full story.
    # Checking the plain name first (as this used to) alerts off the wrong one.
    if 'Churn_Probability_Churn' in merged_results.columns:
        churn_col = 'Churn_Probability_Churn'
    elif 'Churn_Probability' in merged_results.columns:
        churn_col = 'Churn_Probability'
    else:
        return False

    high_risk = merged_results[merged_results[churn_col] > threshold]
    if len(high_risk) < min_customers:
        return False

    if not _should_alert(get_last_alert(company_id, 'churn'), frequency):
        return False

    total_value = high_risk['CLV'].sum() if 'CLV' in high_risk.columns else 0
    top_risk = high_risk.nlargest(10, 'CLV' if 'CLV' in high_risk.columns else churn_col)

    highest_risk_customers = [{
        'id': row.get('Customer ID', ''),
        'name': row.get('Customer Name', f"Customer {row.get('Customer ID', '')}"),
        'churn_risk': row.get(churn_col, 0),
        'clv': row.get('CLV', 0)
    } for _, row in top_risk.iterrows()]

    details = {
        'customer_count': len(high_risk),
        'total_value': total_value,
        'highest_risk_customers': highest_risk_customers
    }

    alert_id = save_alert(company_id, 'churn', float(high_risk[churn_col].mean()), details)
    return alert_id is not None


def check_sales_risk(company_id, sales_data):
    """
    Check for sales risk and record an in-app alert if the threshold is met.

    Args:
        company_id (str): Company ID
        sales_data (pd.DataFrame): Transaction data

    Returns:
        bool: True if an alert was recorded, False otherwise
    """
    try:
        settings = get_alert_settings(company_id, 'sales', SALES_DEFAULTS)
        threshold = settings.get('threshold') or SALES_DEFAULTS['threshold']
        comparison_periods = settings.get('comparison_periods') or SALES_DEFAULTS['comparison_periods']
        frequency = settings.get('alert_frequency') or SALES_DEFAULTS['alert_frequency']

        date_col = next((c for c in sales_data.columns if any(t in c.lower() for t in ['date', 'time', 'day'])), None)
        amount_col = next((c for c in sales_data.columns if any(t in c.lower() for t in ['amount', 'price', 'revenue', 'sales'])), None)

        if not date_col or not amount_col:
            logger.warning("Could not find date or amount column")
            return False

        sales_data = sales_data.copy()
        sales_data[date_col] = pd.to_datetime(sales_data[date_col])

        monthly_sales = sales_data.groupby(pd.Grouper(key=date_col, freq='M'))[amount_col].sum().reset_index()
        monthly_sales = monthly_sales.sort_values(date_col, ascending=False)

        if len(monthly_sales) < comparison_periods + 1:
            logger.info("Not enough data for comparison")
            return False

        current_sales = monthly_sales.iloc[0][amount_col]
        previous_sales = monthly_sales.iloc[1][amount_col]
        decline_percentage = (previous_sales - current_sales) / previous_sales if previous_sales > 0 else 0

        if decline_percentage <= threshold:
            return False

        if not _should_alert(get_last_alert(company_id, 'sales'), frequency):
            return False

        factors = []
        current_month = monthly_sales.iloc[0][date_col].to_pydatetime()
        prev_month = monthly_sales.iloc[1][date_col].to_pydatetime()

        try:
            current_customers = len(sales_data[sales_data[date_col].dt.month == current_month.month]['Customer ID'].unique())
            prev_customers = len(sales_data[sales_data[date_col].dt.month == prev_month.month]['Customer ID'].unique())

            if current_customers < prev_customers:
                customer_decline = (prev_customers - current_customers) / prev_customers
                factors.append(f"Customer count decreased by {customer_decline:.1%} ({prev_customers} to {current_customers})")

            current_aov = current_sales / max(1, current_customers)
            prev_aov = previous_sales / max(1, prev_customers)
            if current_aov < prev_aov:
                aov_decline = (prev_aov - current_aov) / prev_aov
                factors.append(f"Average order value decreased by {aov_decline:.1%} (${prev_aov:.2f} to ${current_aov:.2f})")
        except Exception as e:
            logger.warning(f"Could not calculate customer/AOV factors: {e}")

        if not factors:
            factors.append("General sales decline across customer base")

        details = {
            'current_sales': current_sales,
            'previous_sales': previous_sales,
            'change_amount': previous_sales - current_sales,
            'decline_percentage': decline_percentage,
            'current_month': current_month.strftime('%B %Y'),
            'previous_month': prev_month.strftime('%B %Y'),
            'factors': factors
        }

        alert_id = save_alert(company_id, 'sales', float(decline_percentage), details)
        return alert_id is not None

    except Exception as e:
        logger.error(f"Unexpected error in sales risk check: {e}")
        return False
