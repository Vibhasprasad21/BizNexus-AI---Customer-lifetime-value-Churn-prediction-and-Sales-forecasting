import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

def initialize_firebase_admin():
    """Initialize Firebase Admin SDK for server-side operations like Firestore"""
    if not firebase_admin._apps:
        try:
            # Try using service account details from secrets
            if "firebase" in st.secrets and "service_account" in st.secrets["firebase"]:
                # Create a temporary JSON file using the service account from secrets
                import tempfile
                import json
                
                # Get service account data as a dictionary
                service_account_info = st.secrets["firebase"]["service_account"]
                
                # Create a temporary file to store the service account
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
                    temp_file.write(json.dumps(service_account_info).encode('utf-8'))
                    temp_service_account_path = temp_file.name
                
                # Initialize with the temporary service account file
                cred = credentials.Certificate(temp_service_account_path)
                firebase_admin.initialize_app(cred)
                
                # Clean up temporary file
                os.unlink(temp_service_account_path)
                
                print("Firebase Admin initialized with service account from secrets")
            else:
                print("Service account not found in secrets")
                return None
        except Exception as e:
            print(f"Error initializing Firebase Admin: {e}")
            return None
    
    return firestore.client()

def get_firebase_config():
    """Get Firebase configuration from secrets.toml"""
    if "firebase" in st.secrets and "config" in st.secrets["firebase"]:
        return st.secrets["firebase"]["config"]
    else:
        print("Firebase config not found in secrets")
        return None

def get_pyrebase_config():
    """Get configuration for Pyrebase (Authentication and Storage)"""
    return get_firebase_config()

def get_firebase_auth():
    """Get Firebase Authentication instance"""
    config = get_firebase_config()
    if config:
        firebase = pyrebase.initialize_app(config)
        return firebase.auth()
    return None

def get_firebase_storage():
    """Get Firebase Storage instance"""
    config = get_firebase_config()
    if config:
        firebase = pyrebase.initialize_app(config)
        return firebase.storage()
    return None