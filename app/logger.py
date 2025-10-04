import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] "
    "[%(name)s:%(lineno)d] - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


log_file = os.path.join(LOG_DIR, 'app.log')
logger = logging.getLogger("net4cleanair")
logger.setLevel(logging.DEBUG)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
console_handler.setFormatter(console_formatter)

file_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7, encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
file_handler.setFormatter(file_formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False