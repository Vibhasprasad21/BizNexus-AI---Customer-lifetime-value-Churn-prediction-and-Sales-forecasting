from config.firebase_config import get_firebase_storage
import pandas as pd
import tempfile
import os
import uuid
import joblib

def upload_file(company_id, file_obj, file_name, file_type='data'):
    """
    Upload a file to Firebase Storage
    
    Args:
        company_id (str): Company ID
        file_obj: File object to upload
        file_name (str): Name of the file
        file_type (str): Type of file ('data', 'model', 'report')
        
    Returns:
        dict: Result of the operation with download URL
    """
    try:
        storage = get_firebase_storage()
        
        # Generate a unique file path in storage
        unique_id = str(uuid.uuid4())
        file_path = f"{company_id}/{file_type}/{unique_id}_{file_name}"
        
        # Upload the file
        storage.child(file_path).put(file_obj)
        
        # Get the download URL
        download_url = storage.child(file_path).get_url(None)
        
        return {
            'success': True,
            'file_path': file_path,
            'download_url': download_url
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def download_file(file_path):
    """
    Download a file from Firebase Storage
    
    Args:
        file_path (str): Path to the file in Firebase Storage
        
    Returns:
        dict: Result of the operation with file data
    """
    try:
        storage = get_firebase_storage()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download the file
        storage.child(file_path).download(temp_path)
        
        # Return the path to the downloaded file
        return {
            'success': True,
            'local_path': temp_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def upload_dataframe(company_id, df, file_name='dataset.csv'):
    """
    Upload a pandas DataFrame to Firebase Storage as a CSV file
    
    Args:
        company_id (str): Company ID
        df (pandas.DataFrame): DataFrame to upload
        file_name (str): Name of the file
        
    Returns:
        dict: Result of the operation with download URL
    """
    try:
        # Create a temporary file to store the CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
        
        # Save DataFrame to CSV
        df.to_csv(temp_path, index=False)
        
        # Upload the file
        with open(temp_path, 'rb') as file_obj:
            result = upload_file(company_id, file_obj, file_name, 'data')
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def download_dataframe(file_path):
    """
    Download a CSV file from Firebase Storage as a pandas DataFrame
    
    Args:
        file_path (str): Path to the file in Firebase Storage
        
    Returns:
        dict: Result of the operation with DataFrame
    """
    try:
        # Download the file
        result = download_file(file_path)
        
        if not result['success']:
            return result
        
        # Read the CSV into a DataFrame
        df = pd.read_csv(result['local_path'])
        
        # Clean up the temporary file
        os.unlink(result['local_path'])
        
        return {
            'success': True,
            'dataframe': df
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def upload_model(company_id, model, model_name, model_type='clv'):
    """
    Upload a scikit-learn or other serializable model to Firebase Storage
    
    Args:
        company_id (str): Company ID
        model: Model object to upload
        model_name (str): Name of the model
        model_type (str): Type of model ('clv', 'churn', 'sales')
        
    Returns:
        dict: Result of the operation with download URL
    """
    try:
        # Create a temporary file to store the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as temp_file:
            temp_path = temp_file.name
        
        # Save model using joblib
        joblib.dump(model, temp_path)
        
        # Upload the file
        with open(temp_path, 'rb') as file_obj:
            result = upload_file(company_id, file_obj, f"{model_name}.joblib", f'model/{model_type}')
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def download_model(file_path):
    """
    Download a model from Firebase Storage
    
    Args:
        file_path (str): Path to the model in Firebase Storage
        
    Returns:
        dict: Result of the operation with model object
    """
    try:
        # Download the file
        result = download_file(file_path)
        
        if not result['success']:
            return result
        
        # Load the model
        model = joblib.load(result['local_path'])
        
        # Clean up the temporary file
        os.unlink(result['local_path'])
        
        return {
            'success': True,
            'model': model
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def upload_report(company_id, report_bytes, report_name, report_type='pdf'):
    """
    Upload a report to Firebase Storage
    
    Args:
        company_id (str): Company ID
        report_bytes: Bytes of the report
        report_name (str): Name of the report
        report_type (str): Type of report ('pdf', 'excel')
        
    Returns:
        dict: Result of the operation with download URL
    """
    try:
        # Create a temporary file to store the report
        suffix = '.pdf' if report_type == 'pdf' else '.xlsx'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(report_bytes)
            temp_path = temp_file.name
        
        # Upload the file
        with open(temp_path, 'rb') as file_obj:
            result = upload_file(company_id, file_obj, f"{report_name}{suffix}", 'report')
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }