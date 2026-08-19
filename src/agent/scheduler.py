"""Background/scheduled agent runs, using the `schedule` package.

Streamlit reruns the whole script on every interaction, so any module-level
code here executes on every rerun of every open session. The background
thread is guarded by a process-wide flag (not st.session_state, which is
per-browser-tab and would spawn one thread per tab) so it starts at most
once per server process no matter how many times a page reruns.

Deployment caveat (also surfaced on the AI Agent page): on Streamlit
Community Cloud the app process can be put to sleep or restarted at any
time, silently dropping this thread and its schedule. Community Cloud is
fine for the on-demand "Run Agent Now" button; true unattended background
scheduling needs a host that keeps the process alive continuously (a VM,
container, or `streamlit run` left running on your own machine/server).
SQLite itself is also ephemeral there - see the note on the AI Agent page.
"""
import threading
import time

import schedule

from src.database.db import get_db

_lock = threading.Lock()
_started = False
_thread = None
_last_run_summary = None


def _run_all_companies(trigger='scheduled'):
    global _last_run_summary
    # Imported lazily: Agent pulls in the reasoning backends, which is more
    # to load at module-import time than a background thread needs upfront.
    from src.agent.core import Agent

    agent = Agent()
    with get_db() as conn:
        rows = conn.execute("SELECT DISTINCT company_id FROM datasets").fetchall()

    results = []
    for row in rows:
        company_id = row['company_id']
        try:
            summary = agent.run_cycle(company_id, session_state=None, trigger=trigger)
            results.append((company_id, summary['summary']))
        except Exception as e:
            results.append((company_id, f"error: {e}"))

    _last_run_summary = {'ran_at': time.time(), 'companies': results}


def _worker(interval_minutes):
    schedule.clear('biznexus-agent')
    schedule.every(interval_minutes).minutes.do(_run_all_companies).tag('biznexus-agent')
    while _started:
        schedule.run_pending()
        time.sleep(5)


def start(interval_minutes=60):
    """Start the background scheduler once per server process. Safe to call
    repeatedly - a no-op if already running. Returns True if it just started,
    False if it was already running."""
    global _started, _thread
    with _lock:
        if _started:
            return False
        _started = True
        _thread = threading.Thread(target=_worker, args=(interval_minutes,), daemon=True)
        _thread.start()
        return True


def stop():
    global _started
    with _lock:
        _started = False
        schedule.clear('biznexus-agent')


def is_running():
    return _started


def last_run_summary():
    return _last_run_summary
