import streamlit as st
from config.firebase_config import initialize_firebase_admin
from firebase_admin import firestore
import pandas as pd
import json
import datetime
import uuid

def get_db():
    """Get Firestore database instance"""
    return initialize_firebase_admin()

def save_dataset(company_id, dataset_name, dataset_description, df, user_id):
    """
    Save dataset metadata to Firestore and dataset to Firebase Storage
    
    Args:
        company_id (str): Company ID
        dataset_name (str): Name of the dataset
        dataset_description (str): Description of the dataset
        df (pandas.DataFrame): The dataset as a pandas DataFrame
        user_id (str): ID of the user who uploaded the dataset
        
    Returns:
        dict: Result of the operation
    """
    try:
        db = get_db()
        
        # Create a unique ID for the dataset
        dataset_id = str(uuid.uuid4())
        
        # Convert DataFrame to JSON for storage
        dataset_json = json.loads(df.to_json(orient='records', date_format='iso'))
        
        # Get dataset metadata
        column_info = [{
            "name": col,
            "dtype": str(df[col].dtype),
            "sample_values": df[col].head(3).tolist() if df[col].dtype != 'object' else [str(val) for val in df[col].head(3).tolist()],
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique())
        } for col in df.columns]
        
        # Create dataset metadata document
        dataset_ref = db.collection('companies').document(company_id).collection('datasets').document(dataset_id)
        
        dataset_ref.set({
            'name': dataset_name,
            'description': dataset_description,
            'created_at': firestore.SERVER_TIMESTAMP,
            'created_by': user_id,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': column_info,
            'status': 'active',
            'data': dataset_json,  # Store the actual data in Firestore for simplicity
        })
        
        return {
            'success': True,
            'dataset_id': dataset_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_datasets(company_id):
    """
    Get all datasets for a company
    
    Args:
        company_id (str): Company ID
        
    Returns:
        list: List of dataset metadata
    """
    try:
        db = get_db()
        datasets_ref = db.collection('companies').document(company_id).collection('datasets')
        datasets = datasets_ref.get()
        
        result = []
        for dataset in datasets:
            data = dataset.to_dict()
            # Don't include the actual data in the list to reduce payload size
            if 'data' in data:
                del data['data']
            result.append({
                'id': dataset.id,
                **data
            })
        
        return {
            'success': True,
            'datasets': result
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_dataset(company_id, dataset_id):
    """
    Get a specific dataset
    
    Args:
        company_id (str): Company ID
        dataset_id (str): Dataset ID
        
    Returns:
        dict: Dataset metadata and data
    """
    try:
        db = get_db()
        dataset_ref = db.collection('companies').document(company_id).collection('datasets').document(dataset_id)
        dataset = dataset_ref.get()
        
        if dataset.exists:
            data = dataset.to_dict()
            # Convert data back to DataFrame
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                # Convert DataFrame to dict for return
                data['dataframe'] = df
            
            return {
                'success': True,
                'dataset': {
                    'id': dataset_id,
                    **data
                }
            }
        else:
            return {
                'success': False,
                'error': 'Dataset not found'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def save_analysis_results(company_id, dataset_id, analysis_type, results, model_info=None):
    """
    Save analysis results to Firestore
    
    Args:
        company_id (str): Company ID
        dataset_id (str): Dataset ID
        analysis_type (str): Type of analysis ('clv', 'churn', 'sales')
        results (dict): Analysis results
        model_info (dict, optional): Model information for ML models
        
    Returns:
        dict: Result of the operation
    """
    try:
        db = get_db()
        
        # Create a unique ID for the analysis
        analysis_id = str(uuid.uuid4())
        
        # Reference to the analysis document
        analysis_ref = db.collection('companies').document(company_id) \
            .collection('datasets').document(dataset_id) \
            .collection('analyses').document(analysis_id)
        
        # Prepare data for storage
        analysis_data = {
            'type': analysis_type,
            'created_at': firestore.SERVER_TIMESTAMP,
            'results': results
        }
        
        # Add model info if provided
        if model_info:
            analysis_data['model_info'] = model_info
        
        # Save to Firestore
        analysis_ref.set(analysis_data)
        
        return {
            'success': True,
            'analysis_id': analysis_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_customer_data(company_id, customer_id):
    """
    Get data for a specific customer
    
    Args:
        company_id (str): Company ID
        customer_id (str): Customer ID or identifier
        
    Returns:
        dict: Customer data with analysis results
    """
    try:
        db = get_db()
        
        # Get the latest dataset
        datasets_ref = db.collection('companies').document(company_id).collection('datasets')
        datasets = datasets_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
        
        if not datasets:
            return {
                'success': False,
                'error': 'No datasets found'
            }
        
        dataset = datasets[0]
        dataset_data = dataset.to_dict()
        
        # Convert data to DataFrame
        if 'data' in dataset_data:
            df = pd.DataFrame(dataset_data['data'])
            
            # Find customer data
            # Assuming there's a customer_id column (adjust as needed)
            customer_data = df[df['customer_id'] == customer_id]
            
            if customer_data.empty:
                return {
                    'success': False,
                    'error': f'Customer {customer_id} not found'
                }
            
            # Get analyses for this dataset
            analyses_ref = db.collection('companies').document(company_id) \
                .collection('datasets').document(dataset.id) \
                .collection('analyses')
            
            analyses = analyses_ref.get()
            analyses_data = {}
            
            for analysis in analyses:
                analysis_data = analysis.to_dict()
                analyses_data[analysis_data['type']] = analysis_data['results']
            
            # Return customer data with analyses
            return {
                'success': True,
                'customer': {
                    'id': customer_id,
                    'data': customer_data.to_dict('records')[0],
                    'analyses': analyses_data
                }
            }
        else:
            return {
                'success': False,
                'error': 'Dataset does not contain data'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def save_notification(company_id, notification_type, message, priority='medium', user_id=None):
    """
    Save a notification to Firestore
    
    Args:
        company_id (str): Company ID
        notification_type (str): Type of notification ('alert', 'info', 'warning')
        message (str): Notification message
        priority (str, optional): Priority of the notification ('high', 'medium', 'low')
        user_id (str, optional): User ID if notification is for a specific user
        
    Returns:
        dict: Result of the operation
    """
    try:
        db = get_db()
        
        # Create notification document
        notification_ref = db.collection('companies').document(company_id).collection('notifications').document()
        
        notification_ref.set({
            'type': notification_type,
            'message': message,
            'priority': priority,
            'created_at': firestore.SERVER_TIMESTAMP,
            'read': False,
            'user_id': user_id,
            'for_all_users': user_id is None
        })
        
        return {
            'success': True,
            'notification_id': notification_ref.id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_notifications(company_id, user_id=None, limit=10, unread_only=False):
    """
    Get notifications for a company or user
    
    Args:
        company_id (str): Company ID
        user_id (str, optional): User ID to get user-specific notifications
        limit (int, optional): Maximum number of notifications to return
        unread_only (bool, optional): Get only unread notifications
        
    Returns:
        dict: Notifications data
    """
    try:
        db = get_db()
        notifications_ref = db.collection('companies').document(company_id).collection('notifications')
        
        # Base query - order by creation time
        query = notifications_ref.order_by('created_at', direction=firestore.Query.DESCENDING)
        
        # Add filters
        if unread_only:
            query = query.where('read', '==', False)
        
        if user_id:
            # Get notifications for all users OR specifically for this user
            query = query.where('for_all_users', '==', True) #.or_('user_id', '==', user_id)
        
        # Limit results
        query = query.limit(limit)
        
        # Execute query
        notifications = query.get()
        
        result = []
        for notification in notifications:
            data = notification.to_dict()
            result.append({
                'id': notification.id,
                **data,
                # Convert timestamp to string for JSON serialization
                'created_at': data['created_at'].strftime('%Y-%m-%d %H:%M:%S') if 'created_at' in data else None
            })
        
        return {
            'success': True,
            'notifications': result
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def mark_notification_read(company_id, notification_id):
    """
    Mark a notification as read
    
    Args:
        company_id (str): Company ID
        notification_id (str): Notification ID
        
    Returns:
        dict: Result of the operation
    """
    try:
        db = get_db()
        notification_ref = db.collection('companies').document(company_id).collection('notifications').document(notification_id)
        
        notification_ref.update({
            'read': True,
            'read_at': firestore.SERVER_TIMESTAMP
        })
        
        return {
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }