import os
from dotenv import load_dotenv

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

# -----------------------
# Base Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------
# Model Paths
# -----------------------
MODEL_HYBRID_PATH = os.getenv(
    "MODEL_HYBRID_PATH",
    os.path.join(BASE_DIR, "models", "hybrid_best.h5")
)

MODEL_CNN_PATH = os.getenv(
    "MODEL_CNN_PATH",
    os.path.join(BASE_DIR, "models", "cnn_best.h5")
)

# Path to model metadata (exported from training pipeline)
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# -----------------------
# Upload Config
# -----------------------
UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER",
    os.path.join(BASE_DIR, "uploads", "tmp")
)

# Maximum upload size (default = 10 MB)
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))

# Allowed image file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# -----------------------
# Inference & Threshold Config
# -----------------------
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.7))

# Image preprocessing target size
IMAGE_SIZE = (
    int(os.getenv("IMAGE_WIDTH", 128)),   # width first for Pillow consistency
    int(os.getenv("IMAGE_HEIGHT", 128))
)

# -----------------------
# Metadata
# -----------------------
API_VERSION = "1.0.0"
PROJECT_NAME = "NeuroSense – Epileptic Seizure Prediction API"

# -----------------------
# Database (MongoDB)
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/neurosense")

# -----------------------
# JWT Authentication
# -----------------------
JWT_SECRET = os.getenv("JWT_SECRET", "defaultsecretkey")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", 24))
