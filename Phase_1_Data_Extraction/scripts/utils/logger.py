import logging
import sys
from datetime import datetime
from pathlib import Path
import json

def setup_logger(name: str, level: str = "INFO", log_file: str = None):
    """Setup logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def log_experiment(logger, config, results):
    """Log experiment results"""
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results
    }
    
    # Create results directory
    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"experiment_{timestamp}.json"
    
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    
    logger.info(f"Experiment results saved to {log_file}")
    return log_file