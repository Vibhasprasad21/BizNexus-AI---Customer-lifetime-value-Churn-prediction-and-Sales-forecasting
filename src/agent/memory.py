"""Persistence for the agent's own reasoning trail.

This is the agent's "memory" in the literal sense used by the rest of the
codebase's store.py: plain functions over SQLite, one connection per call.
Every reasoning cycle, the signals it noticed, the tools it called, the
recommendations it logged, and (later) whether those recommendations turned
out to be right all live here - so a future cycle, or a human looking at the
Agent Activity Log page, can see why the agent did what it did.
"""
import json
import uuid
from datetime import datetime

from src.database.db import get_db


# --- Cycles --------------------------------------------------------------

def start_cycle(company_id, trigger, reasoning_backend='rule_based'):
    """Record the start of a new perceive->reason->plan->act->observe cycle."""
    cycle_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO agent_cycles
               (id, company_id, trigger, started_at, status, reasoning_backend)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (cycle_id, company_id, trigger, datetime.now().isoformat(), reasoning_backend)
        )
    return cycle_id


def finish_cycle(cycle_id, signals_found, actions_taken, summary):
    """Close out a cycle with a final tally and a one-line human summary."""
    with get_db() as conn:
        conn.execute(
            """UPDATE agent_cycles SET finished_at = ?, status = 'complete',
               signals_found = ?, actions_taken = ?, summary = ? WHERE id = ?""",
            (datetime.now().isoformat(), signals_found, actions_taken, summary, cycle_id)
        )


def get_cycles(company_id, limit=20):
    """List recent agent cycles for a company, most recent first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM agent_cycles WHERE company_id = ?
               ORDER BY started_at DESC LIMIT ?""",
            (company_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def get_last_cycle(company_id):
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM agent_cycles WHERE company_id = ? AND status = 'complete'
               ORDER BY started_at DESC LIMIT 1""",
            (company_id,)
        ).fetchone()
    return dict(row) if row else None


# --- Events (the reasoning trail) ----------------------------------------

def log_event(cycle_id, company_id, step, message, signal_type=None, data=None):
    """Append one step to the reasoning trail.

    `step` is one of 'perceive' | 'reason' | 'plan' | 'act' | 'observe'.
    `message` is the human-readable sentence shown on the Agent page, e.g.
    "Noticed 12 high-CLV customers crossed into high churn risk".
    """
    event_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO agent_events
               (id, cycle_id, company_id, step, signal_type, message, data_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, cycle_id, company_id, step, signal_type, message,
             json.dumps(data, default=str) if data is not None else None,
             datetime.now().isoformat())
        )
    return event_id


def get_events_for_cycle(cycle_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_events WHERE cycle_id = ? ORDER BY created_at ASC",
            (cycle_id,)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d['data'] = json.loads(d.pop('data_json') or 'null')
        result.append(d)
    return result


def get_recent_events(company_id, limit=100):
    """Flat, most-recent-first trail across cycles, for the activity log page."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM agent_events WHERE company_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (company_id, limit)
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d['data'] = json.loads(d.pop('data_json') or 'null')
        result.append(d)
    return result


# --- Recommendations (pending actions a human can approve/dismiss) -------

def log_recommendation(company_id, title, body, cycle_id=None, signal_type=None,
                        target_customers=None, priority='medium'):
    rec_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO agent_recommendations
               (id, cycle_id, company_id, signal_type, title, body,
                target_customers_json, priority, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (rec_id, cycle_id, company_id, signal_type, title, body,
             json.dumps(target_customers, default=str) if target_customers else None,
             priority, datetime.now().isoformat())
        )
    return rec_id


def get_recommendations(company_id, status=None, limit=50):
    query = "SELECT * FROM agent_recommendations WHERE company_id = ?"
    params = [company_id]
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d['target_customers'] = json.loads(d.pop('target_customers_json') or '[]')
        result.append(d)
    return result


def has_recent_recommendation(company_id, signal_type, hours=24):
    """Whether a recommendation of this signal type was already logged
    within the cooldown window. Used to throttle signals that don't have a
    natural "what's new" diff (revenue dip, CLV drop, seasonal anomaly) so
    the agent doesn't re-alert on the same standing condition every cycle."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT created_at FROM agent_recommendations
               WHERE company_id = ? AND signal_type = ?
               ORDER BY created_at DESC LIMIT 1""",
            (company_id, signal_type)
        ).fetchone()
    if not row:
        return False
    elapsed = (datetime.now() - datetime.fromisoformat(row['created_at'])).total_seconds()
    return elapsed < hours * 3600


def get_last_cycle_recommendations_by_signal(company_id, signal_type):
    """All recommendations of a signal type from the most recent cycle that
    produced any. A single signal firing can log more than one recommendation
    (e.g. one per customer tier) - diffing against just the single latest row
    would silently drop the customers from every recommendation logged
    earlier in that same cycle, so this pulls the whole cycle's worth."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT cycle_id FROM agent_recommendations
               WHERE company_id = ? AND signal_type = ? AND cycle_id IS NOT NULL
               ORDER BY created_at DESC LIMIT 1""",
            (company_id, signal_type)
        ).fetchone()
        if not row:
            return []

        rows = conn.execute(
            """SELECT * FROM agent_recommendations
               WHERE company_id = ? AND signal_type = ? AND cycle_id = ?""",
            (company_id, signal_type, row['cycle_id'])
        ).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d['target_customers'] = json.loads(d.pop('target_customers_json') or '[]')
        result.append(d)
    return result


def decide_recommendation(rec_id, status, decided_by=None):
    """Approve or dismiss a pending recommendation ('approved' | 'dismissed')."""
    with get_db() as conn:
        conn.execute(
            """UPDATE agent_recommendations SET status = ?, decided_at = ?, decided_by = ?
               WHERE id = ?""",
            (status, datetime.now().isoformat(), decided_by, rec_id)
        )


def record_outcome(rec_id, outcome, notes=None):
    """Called during a later OBSERVE step: was this recommendation right?"""
    with get_db() as conn:
        conn.execute(
            """UPDATE agent_recommendations SET outcome = ?, outcome_notes = ?,
               outcome_recorded_at = ? WHERE id = ?""",
            (outcome, notes, datetime.now().isoformat(), rec_id)
        )


def get_recommendations_needing_observation(company_id, signal_type=None, min_age_hours=0):
    """Past recommendations without a recorded outcome yet - candidates for OBSERVE."""
    with get_db() as conn:
        query = """SELECT * FROM agent_recommendations
                   WHERE company_id = ? AND outcome IS NULL AND status != 'dismissed'"""
        params = [company_id]
        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)
        query += " ORDER BY created_at ASC"
        rows = conn.execute(query, params).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        created = datetime.fromisoformat(d['created_at'])
        if (datetime.now() - created).total_seconds() < min_age_hours * 3600:
            continue
        d['target_customers'] = json.loads(d.pop('target_customers_json') or '[]')
        result.append(d)
    return result


# --- Metric snapshots (lets a fresh session still detect "compared to last time") ---

def save_snapshot(company_id, metric, value):
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO agent_snapshots (company_id, metric, value, recorded_at)
               VALUES (?, ?, ?, ?)""",
            (company_id, metric, value, datetime.now().isoformat())
        )


def get_last_snapshot(company_id, metric, before=None):
    """Most recent snapshot for a metric, optionally strictly before a given cycle's start."""
    with get_db() as conn:
        query = "SELECT * FROM agent_snapshots WHERE company_id = ? AND metric = ?"
        params = [company_id, metric]
        if before:
            query += " AND recorded_at < ?"
            params.append(before)
        query += " ORDER BY recorded_at DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None
