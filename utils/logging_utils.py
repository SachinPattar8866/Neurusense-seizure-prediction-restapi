import logging
import os

# Ensure logs directory exists
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure basic logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log_request(req):
    """Log incoming request details."""
    try:
        endpoint = req.path
        method = req.method
        logging.info(f"[REQUEST] {method} {endpoint}")
    except Exception:
        pass

def log_response(endpoint: str, response: dict):
    """Log outgoing response."""
    logging.info(f"[RESPONSE] {endpoint} -> {response}")

def log_error(endpoint: str, error: str):
    """Log an error event."""
    logging.error(f"[ERROR] {endpoint} -> {error}")
