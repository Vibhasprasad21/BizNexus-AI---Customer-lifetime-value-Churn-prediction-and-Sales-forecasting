import os
import sqlite3
from contextlib import contextmanager

DB_DIR = os.path.join(os.getcwd(), 'data')
DB_PATH = os.path.join(DB_DIR, 'biznexus.db')

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT,
    company_id TEXT,
    role TEXT DEFAULT 'analyst',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    name TEXT,
    description TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    row_count INTEGER,
    column_count INTEGER,
    columns_json TEXT,
    data_json TEXT,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    dataset_id TEXT,
    type TEXT,
    created_at TEXT NOT NULL,
    results_json TEXT,
    model_info_json TEXT
);

CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    type TEXT,
    message TEXT,
    priority TEXT DEFAULT 'medium',
    created_at TEXT NOT NULL,
    read INTEGER DEFAULT 0,
    user_id TEXT,
    for_all_users INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    type TEXT,
    risk_level REAL,
    details_json TEXT,
    status TEXT DEFAULT 'delivered',
    created_at TEXT NOT NULL,
    delivered_at TEXT
);

CREATE TABLE IF NOT EXISTS alert_settings (
    company_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    threshold REAL,
    min_customers INTEGER,
    comparison_periods INTEGER,
    alert_frequency TEXT DEFAULT 'daily',
    PRIMARY KEY (company_id, alert_type)
);

CREATE TABLE IF NOT EXISTS agent_cycles (
    id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT DEFAULT 'running',
    signals_found INTEGER DEFAULT 0,
    actions_taken INTEGER DEFAULT 0,
    summary TEXT,
    reasoning_backend TEXT DEFAULT 'rule_based'
);

CREATE TABLE IF NOT EXISTS agent_events (
    id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    step TEXT NOT NULL,
    signal_type TEXT,
    message TEXT NOT NULL,
    data_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_recommendations (
    id TEXT PRIMARY KEY,
    cycle_id TEXT,
    company_id TEXT NOT NULL,
    signal_type TEXT,
    title TEXT NOT NULL,
    body TEXT,
    target_customers_json TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decided_by TEXT,
    outcome TEXT,
    outcome_notes TEXT,
    outcome_recorded_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_snapshots (
    company_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (company_id, metric, recorded_at)
);
"""


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db():
    """Yield a short-lived SQLite connection with row access by column name.

    A fresh connection per call avoids cross-thread reuse issues under
    Streamlit's script-rerun model, and SQLite handles that cheaply.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


init_db()
