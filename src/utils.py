import pickle
import os
import sys
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

def save_object(file_path, obj):
    """Save Python object to pickle file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load Python object from pickle file"""
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Loaded object from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
