import re
import uuid
from datetime import datetime

import bcrypt
import streamlit as st

from src.database.db import get_db


def is_valid_email(email):
    """Check if email is valid"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None


def register_user(email, password, full_name, company_name):
    """Register a new user, hashing the password locally with bcrypt."""
    try:
        with get_db() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if existing:
                return {'success': False, 'error': 'An account with this email already exists'}

            company_row = conn.execute(
                "SELECT id FROM companies WHERE name = ?", (company_name,)
            ).fetchone()

            now = datetime.now().isoformat()
            if company_row:
                company_id = company_row['id']
            else:
                company_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO companies (id, name, created_at) VALUES (?, ?, ?)",
                    (company_id, company_name, now)
                )

            user_id = str(uuid.uuid4())
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

            conn.execute(
                """INSERT INTO users (id, email, password_hash, full_name, company_id, role, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, email, password_hash, full_name, company_id, 'analyst', now)
            )

        return {'success': True, 'user_id': user_id, 'company_id': company_id}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def login_user(email, password):
    """Verify credentials against the local SQLite user table."""
    try:
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

            if not user:
                return {'success': False, 'error': 'No account found with this email'}

            if not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
                return {'success': False, 'error': 'Incorrect password'}

            company = conn.execute(
                "SELECT name FROM companies WHERE id = ?", (user['company_id'],)
            ).fetchone()
            company_name = company['name'] if company else 'Unknown Company'

        return {
            'success': True,
            'user_id': user['id'],
            'email': user['email'],
            'full_name': user['full_name'],
            'company_id': user['company_id'],
            'company_name': company_name,
            'role': user['role'] or 'analyst',
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def logout_user():
    """Clear user session state"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.company_id = None
    return True


def get_user_data(user_id):
    """Get user data by ID"""
    try:
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, email, full_name, company_id, role, created_at FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()

        if user:
            return {'success': True, 'data': dict(user)}
        return {'success': False, 'error': 'User not found'}

    except Exception as e:
        return {'success': False, 'error': str(e)}
