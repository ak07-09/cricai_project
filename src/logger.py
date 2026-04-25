import logging
import os
from datetime import datetime

# Create logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Logging configuration
logging.basicConfig(
    filename=log_file,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG
)

def get_logger(name):
    return logging.getLogger(name)
