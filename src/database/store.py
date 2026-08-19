import json
import uuid
from datetime import datetime

import pandas as pd

from src.database.db import get_db


def save_dataset(company_id, dataset_name, dataset_description, df, user_id):
    """Save a dataset (metadata + rows) to the local database."""
    try:
        dataset_id = str(uuid.uuid4())
        dataset_json = json.loads(df.to_json(orient='records', date_format='iso'))

        column_info = [{
            "name": col,
            "dtype": str(df[col].dtype),
            "sample_values": (
                df[col].head(3).tolist()
                if df[col].dtype != 'object' and not pd.api.types.is_datetime64_any_dtype(df[col])
                else [str(val) for val in df[col].head(3).tolist()]
            ),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique())
        } for col in df.columns]

        with get_db() as conn:
            conn.execute(
                """INSERT INTO datasets
                   (id, company_id, name, description, created_at, created_by,
                    row_count, column_count, columns_json, data_json, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
                (dataset_id, company_id, dataset_name, dataset_description,
                 datetime.now().isoformat(), user_id, len(df), len(df.columns),
                 json.dumps(column_info), json.dumps(dataset_json))
            )

        return {'success': True, 'dataset_id': dataset_id}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_datasets(company_id):
    """List datasets for a company (without the row data, to keep it light)."""
    try:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT id, name, description, created_at, created_by,
                          row_count, column_count, columns_json, status
                   FROM datasets WHERE company_id = ? ORDER BY created_at DESC""",
                (company_id,)
            ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d['columns'] = json.loads(d.pop('columns_json') or '[]')
            result.append(d)

        return {'success': True, 'datasets': result}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_dataset(company_id, dataset_id):
    """Get a specific dataset including its data as a DataFrame."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE company_id = ? AND id = ?",
                (company_id, dataset_id)
            ).fetchone()

        if not row:
            return {'success': False, 'error': 'Dataset not found'}

        data = dict(row)
        records = json.loads(data.pop('data_json') or '[]')
        data['columns'] = json.loads(data.pop('columns_json') or '[]')
        data['dataframe'] = pd.DataFrame(records)

        return {'success': True, 'dataset': {'id': dataset_id, **data}}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def save_analysis_results(company_id, dataset_id, analysis_type, results, model_info=None):
    """Save analysis results (CLV/churn/sales) linked to a dataset."""
    try:
        analysis_id = str(uuid.uuid4())

        with get_db() as conn:
            conn.execute(
                """INSERT INTO analyses (id, company_id, dataset_id, type, created_at, results_json, model_info_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (analysis_id, company_id, dataset_id, analysis_type, datetime.now().isoformat(),
                 json.dumps(results, default=str), json.dumps(model_info, default=str) if model_info else None)
            )

        return {'success': True, 'analysis_id': analysis_id}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_latest_analysis(company_id, analysis_type):
    """Get the most recently saved analysis of a given type (e.g. 'clv',
    'churn', 'sales_forecast'), regardless of which dataset it came from.

    Used by the agent when it runs without a live Streamlit session (e.g. a
    scheduled background cycle) and has to reconstruct state from the DB
    instead of st.session_state.
    """
    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT * FROM analyses WHERE company_id = ? AND type = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (company_id, analysis_type)
            ).fetchone()

        if not row:
            return None

        d = dict(row)
        d['results'] = json.loads(d.pop('results_json') or 'null')
        d['model_info'] = json.loads(d.pop('model_info_json') or 'null')
        return d

    except Exception:
        return None


def get_customer_data(company_id, customer_id):
    """Get a customer's data plus any analyses from the most recent dataset."""
    try:
        with get_db() as conn:
            dataset_row = conn.execute(
                """SELECT id, data_json FROM datasets WHERE company_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (company_id,)
            ).fetchone()

            if not dataset_row:
                return {'success': False, 'error': 'No datasets found'}

            records = json.loads(dataset_row['data_json'] or '[]')
            df = pd.DataFrame(records)

            if 'customer_id' not in df.columns:
                return {'success': False, 'error': 'Dataset does not contain a customer_id column'}

            customer_data = df[df['customer_id'] == customer_id]
            if customer_data.empty:
                return {'success': False, 'error': f'Customer {customer_id} not found'}

            analyses_rows = conn.execute(
                "SELECT type, results_json FROM analyses WHERE company_id = ? AND dataset_id = ?",
                (company_id, dataset_row['id'])
            ).fetchall()

        analyses_data = {row['type']: json.loads(row['results_json']) for row in analyses_rows}

        return {
            'success': True,
            'customer': {
                'id': customer_id,
                'data': customer_data.to_dict('records')[0],
                'analyses': analyses_data
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


# --- Notifications -----------------------------------------------------

def save_notification(company_id, notification_type, message, priority='medium', user_id=None):
    """Save an in-app notification."""
    try:
        notification_id = str(uuid.uuid4())

        with get_db() as conn:
            conn.execute(
                """INSERT INTO notifications (id, company_id, type, message, priority, created_at, read, user_id, for_all_users)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (notification_id, company_id, notification_type, message, priority,
                 datetime.now().isoformat(), user_id, 1 if user_id is None else 0)
            )

        return {'success': True, 'notification_id': notification_id}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_notifications(company_id, user_id=None, limit=10, unread_only=False):
    """Get notifications for a company, most recent first."""
    try:
        query = "SELECT * FROM notifications WHERE company_id = ?"
        params = [company_id]

        if unread_only:
            query += " AND read = 0"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with get_db() as conn:
            rows = conn.execute(query, params).fetchall()

        result = [dict(row) for row in rows]
        return {'success': True, 'notifications': result}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def mark_notification_read(company_id, notification_id):
    """Mark a notification as read."""
    try:
        with get_db() as conn:
            conn.execute(
                "UPDATE notifications SET read = 1 WHERE company_id = ? AND id = ?",
                (company_id, notification_id)
            )
        return {'success': True}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def count_unread_notifications(company_id):
    """Count unread notifications for the sidebar badge."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM notifications WHERE company_id = ? AND read = 0",
                (company_id,)
            ).fetchone()
        return row['c'] if row else 0
    except Exception:
        return 0


# --- Risk alerts ---------------------------------------------------------

def get_alert_settings(company_id, alert_type, defaults):
    """Get alert settings for a company/type, creating defaults if missing."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM alert_settings WHERE company_id = ? AND alert_type = ?",
                (company_id, alert_type)
            ).fetchone()

            if row:
                return dict(row)

            conn.execute(
                """INSERT INTO alert_settings (company_id, alert_type, threshold, min_customers, comparison_periods, alert_frequency)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (company_id, alert_type, defaults.get('threshold'), defaults.get('min_customers'),
                 defaults.get('comparison_periods'), defaults.get('alert_frequency', 'daily'))
            )

        return {'company_id': company_id, 'alert_type': alert_type, **defaults}

    except Exception:
        return {'company_id': company_id, 'alert_type': alert_type, **defaults}


def save_alert_settings(company_id, alert_type, threshold=None, min_customers=None,
                         comparison_periods=None, alert_frequency=None):
    """Update alert settings for a company/type."""
    try:
        with get_db() as conn:
            existing = conn.execute(
                "SELECT 1 FROM alert_settings WHERE company_id = ? AND alert_type = ?",
                (company_id, alert_type)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE alert_settings SET threshold = ?, min_customers = ?,
                       comparison_periods = ?, alert_frequency = ?
                       WHERE company_id = ? AND alert_type = ?""",
                    (threshold, min_customers, comparison_periods, alert_frequency, company_id, alert_type)
                )
            else:
                conn.execute(
                    """INSERT INTO alert_settings
                       (company_id, alert_type, threshold, min_customers, comparison_periods, alert_frequency)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (company_id, alert_type, threshold, min_customers, comparison_periods, alert_frequency)
                )
        return True
    except Exception:
        return False


def save_alert(company_id, alert_type, risk_level, details):
    """Persist a triggered risk alert as an in-app record (delivered immediately)."""
    try:
        alert_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with get_db() as conn:
            conn.execute(
                """INSERT INTO alerts (id, company_id, type, risk_level, details_json, status, created_at, delivered_at)
                   VALUES (?, ?, ?, ?, ?, 'delivered', ?, ?)""",
                (alert_id, company_id, alert_type, risk_level, json.dumps(details, default=str), now, now)
            )

        save_notification(
            company_id,
            notification_type=alert_type,
            message=_alert_summary(alert_type, details),
            priority='high'
        )

        return alert_id

    except Exception:
        return None


def _alert_summary(alert_type, details):
    if alert_type == 'churn':
        return (f"High churn risk: {details.get('customer_count', 0)} customers, "
                f"${details.get('total_value', 0):,.2f} at risk")
    return (f"Sales decline: {details.get('decline_percentage', 0):.1%} drop "
            f"(-${details.get('change_amount', 0):,.2f})")


def get_last_alert(company_id, alert_type):
    """Get the most recent alert of a given type, for frequency throttling."""
    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT * FROM alerts WHERE company_id = ? AND type = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (company_id, alert_type)
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def get_alerts(company_id, alert_type=None, since=None):
    """List alerts for a company, most recent first."""
    try:
        query = "SELECT * FROM alerts WHERE company_id = ?"
        params = [company_id]

        if alert_type:
            query += " AND type = ?"
            params.append(alert_type)

        if since:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        query += " ORDER BY created_at DESC"

        with get_db() as conn:
            rows = conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d['details'] = json.loads(d.pop('details_json') or '{}')
            result.append(d)

        return result

    except Exception:
        return []
