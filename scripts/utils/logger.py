import logging
import json
import sys
from pythonjsonlogger import jsonlogger
from pathlib import Path
import yaml

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

def setup_logger(name, log_dir="logs", level=logging.INFO):
    """Setup JSON logger for the project"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # JSON file handler
    json_handler = logging.FileHandler(log_dir / f"{name}.jsonl")
    json_formatter = CustomJsonFormatter(
        '%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s'
    )
    json_handler.setFormatter(json_formatter)
    
    # Console handler (readable format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(json_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_experiment(logger, config, results, model_info=None):
    """Log experiment details and results"""
    experiment_log = {
        "experiment_type": "text_modeling",
        "config": config,
        "results": results,
        "model_info": model_info,
        "timestamp": logging.Formatter().formatTime(logging.LogRecord(
            name=logger.name, level=logging.INFO, pathname="", lineno=0, msg=""
        ))
    }
    
    logger.info("Experiment completed", extra={"experiment": experiment_log})
    
    # Also save to separate file
    experiment_dir = Path("experiments") / "text_models"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_file = experiment_dir / f"experiment_{results.get('timestamp', 'unknown')}.json"
    with open(experiment_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    return experiment_file