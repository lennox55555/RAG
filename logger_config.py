import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# create out dir if needed
out_dir = Path("out")
out_dir.mkdir(exist_ok=True)

def setup_logger(name):
    """set up a logger with file and console output"""
    logger = logging.getLogger(name)
    
    # only configure if no handlers yet
    if not logger.handlers:
        # set log level
        logger.setLevel(logging.INFO)
        
        # create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # create file handler
        log_file = out_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger