"""
Central logging setup for the whole service.
 
Every other module gets its logger by calling get_logger(__name__) --
nobody configures logging by itself, and nobody uses print(). This module
sets the log level, the message format, and sends every log record to
both the console and a log file, using settings from config.yaml.
"""
 
import logging
import sys
from pathlib import Path
 
from src.config import CONFIG, PROJECT_ROOT
 
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
 
_configured = False  # guards against setting up handlers more than once
 
 
def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return
 
    log_cfg = CONFIG["logging"]
    level = getattr(logging, log_cfg["level"].upper(), logging.INFO)
    log_file_path = PROJECT_ROOT / log_cfg["log_file"]
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
 
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
 
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
 
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
 
    _configured = True
 
 
def get_logger(name: str) -> logging.Logger:
    """
    Every module should call: logger = get_logger(__name__)
    This guarantees logging is configured exactly once, no matter which
    module happens to import it first.
    """
    _configure_root_logger()
    return logging.getLogger(name)
 
 
if __name__ == "__main__":
    # Manual check: "python -m src.logging_config" from the project root (task3/)
    logger = get_logger(__name__)
    logger.debug("This is a debug message (won't show at INFO level).")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    print(f"Check the log file at: {PROJECT_ROOT / CONFIG['logging']['log_file']}")
 


























