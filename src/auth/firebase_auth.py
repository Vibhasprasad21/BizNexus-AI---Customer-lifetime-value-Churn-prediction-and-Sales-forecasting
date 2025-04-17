import streamlit as st
from config.firebase_config import get_firebase_auth, initialize_firebase_admin
import firebase_admin.auth
from firebase_admin import firestore
import re
# In src/auth/firebase_auth.py

import base64
from cryptography.fernet import Fernet
from firebase_admin import firestore

# Add these utility functions to the existing file
def encrypt_sensitive_data(data, key=None):
    """
    Encrypt sensitive data like passwords
    
    Args:
        data (str): Data to encrypt
        key (bytes, optional): Encryption key. Generates a new key if not provided.
    
    Returns:
        dict: Containing encryption key and encrypted data
    """
    # Generate a new key if not provided
    if key is None:
        key = Fernet.generate_key()
    
    cipher_suite = Fernet(key)
    
    # Encrypt the data
    encrypted_data = cipher_suite.encrypt(data.encode())
    
    return {
        'key': base64.urlsafe_b64encode(key).decode(),
        'encrypted_data': base64.urlsafe_b64encode(encrypted_data).decode()
    }

def decrypt_sensitive_data(encrypted_data, key):
    """
    Decrypt previously encrypted data
    
    Args:
        encrypted_data (str): Base64 encoded encrypted data
        key (str): Base64 encoded encryption key
    
    Returns:
        str: Decrypted data, or None if decryption fails
    """
    try:
        # Decode the key and encrypted data
        decoded_key = base64.urlsafe_b64decode(key)
        decoded_encrypted_data = base64.urlsafe_b64decode(encrypted_data)
        
        # Create cipher suite and decrypt
        cipher_suite = Fernet(decoded_key)
        decrypted_data = cipher_suite.decrypt(decoded_encrypted_data)
        
        return decrypted_data.decode()
    
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

def update_user_smtp_config(db, user_id, smtp_username, smtp_password):
    """
    Update user's SMTP configuration with encrypted credentials
    
    Args:
        db (firestore.Client): Firestore database client
        user_id (str): Firebase user ID
        smtp_username (str): SMTP username
        smtp_password (str): SMTP password
    
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        # Encrypt the SMTP password
        encrypted_credentials = encrypt_sensitive_data(smtp_password)
        
        # Store in Firestore
        db.collection('users').document(user_id).update({
            'email_config': {
                'smtp_username': smtp_username,
                'encrypted_password': encrypted_credentials['encrypted_data'],
                'encryption_key': encrypted_credentials['key'],
                'configured': True
            }
        })
        
        return True
    except Exception as e:
        print(f"Error updating SMTP configuration: {e}")
        return False

def retrieve_user_smtp_config(db, user_id):
    """
    Retrieve and decrypt user's SMTP configuration
    
    Args:
        db (firestore.Client): Firestore database client
        user_id (str): Firebase user ID
    
    Returns:
        dict: SMTP configuration or None if not found/configured
    """
    try:
        user_doc = db.collection('users').document(user_id).get()
        
        if user_doc.exists:
            email_config = user_doc.to_dict().get('email_config', {})
            
            # Check if email is configured
            if not email_config.get('configured', False):
                return None
            
            # Decrypt password
            decrypted_password = decrypt_sensitive_data(
                email_config.get('encrypted_password'),
                email_config.get('encryption_key')
            )
            
            return {
                'smtp_username': email_config.get('smtp_username'),
                'smtp_password': decrypted_password,
                'smtp_host': email_config.get('smtp_host', 'smtp.gmail.com'),
                'smtp_port': email_config.get('smtp_port', 587)
            }
        
        return None
    
    except Exception as e:
        print(f"Error retrieving SMTP configuration: {e}")
        return None

# Modify the existing login_user function to include email configuration check
def login_user(email, password):
    """Login user with Firebase Authentication"""
    try:
        print(f"Attempting to login user: {email}")
        
        # Initialize Firestore first to ensure admin SDK is initialized
        db = initialize_firebase_admin()
        print("Firebase Admin SDK initialized successfully")
        
        # Get auth instance
        auth = get_firebase_auth()
        print("Firebase Auth instance obtained")
        
        # Attempt login
        user = auth.sign_in_with_email_and_password(email, password)
        print(f"User authenticated successfully: {user['localId']}")
        
        # Get user information from Firestore
        user_doc = db.collection('users').document(user['localId']).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            company_id = user_data.get('company_id')
            
            # Get company information
            company_doc = db.collection('companies').document(company_id).get()
            company_name = company_doc.to_dict().get('name') if company_doc.exists else "Unknown Company"
            
            print(f"User profile retrieved: {user_data.get('full_name')}")
            
            return {
                'success': True,
                'user_id': user['localId'],
                'email': email,
                'full_name': user_data.get('full_name'),
                'company_id': company_id,
                'company_name': company_name,
                'role': user_data.get('role', 'analyst'),
                'token': user['idToken']
            }
        else:
            print("User profile not found in Firestore")
            return {
                'success': False,
                'error': 'User profile not found'
            }
            
    except Exception as e:
        print(f"Login error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def is_valid_email(email):
    """Check if email is valid"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def register_user(email, password, full_name, company_name):
    """Register a new user in Firebase Authentication and Firestore"""
    try:
        # Get auth and firestore instances
        auth = get_firebase_auth()
        db = initialize_firebase_admin()
        
        # Create user in Firebase Authentication
        user = auth.create_user_with_email_and_password(email, password)
        user_id = user['localId']
        
        # Check if company exists in Firestore
        companies_ref = db.collection('companies')
        company_query = companies_ref.where('name', '==', company_name).limit(1).get()
        users_ref.document(user_id).update({
            'email_config': {
                'smtp_host': '',  # Optional: user can configure later
                'smtp_port': 587,
                'smtp_username': '',
                'smtp_password': '',  # Securely store encrypted
                'use_tls': True
            }
        })
        
        if len(company_query) == 0:
            # Create new company
            company_ref = companies_ref.document()
            company_id = company_ref.id
            company_ref.set({
                'name': company_name,
                'created_at': firestore.SERVER_TIMESTAMP,
                'admin_user_id': user_id
            })
        else:
            # Use existing company
            company_id = company_query[0].id
        
        # Add user details to Firestore
        users_ref = db.collection('users')
        users_ref.document(user_id).set({
            'email': email,
            'full_name': full_name,
            'company_id': company_id,
            'role': 'analyst',
            'created_at': firestore.SERVER_TIMESTAMP
        })
        
        # Add user to company's analysts
        db.collection('companies').document(company_id).collection('analysts').document(user_id).set({
            'email': email,
            'full_name': full_name,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        
        return {
            'success': True,
            'user_id': user_id,
            'company_id': company_id
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def login_user(email, password):
    """Login user with Firebase Authentication"""
    try:
        auth = get_firebase_auth()
        user = auth.sign_in_with_email_and_password(email, password)
        
        # Get user information from Firestore
        db = initialize_firebase_admin()
        user_doc = db.collection('users').document(user['localId']).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            company_id = user_data.get('company_id')
            
            # Get company information
            company_doc = db.collection('companies').document(company_id).get()
            company_name = company_doc.to_dict().get('name') if company_doc.exists else "Unknown Company"
            
            return {
                'success': True,
                'user_id': user['localId'],
                'email': email,
                'full_name': user_data.get('full_name'),
                'company_id': company_id,
                'company_name': company_name,
                'role': user_data.get('role', 'analyst'),
                'token': user['idToken']
            }
        else:
            return {
                'success': False,
                'error': 'User profile not found'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def logout_user():
    """Clear user session state"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.company_id = None
    return True

def get_user_data(user_id):
    """Get user data from Firestore"""
    try:
        db = initialize_firebase_admin()
        user_doc = db.collection('users').document(user_id).get()
        
        if user_doc.exists:
            return {
                'success': True,
                'data': user_doc.to_dict()
            }
        else:
            return {
                'success': False,
                'error': 'User not found'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }