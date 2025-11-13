import os
import uuid
from werkzeug.utils import secure_filename
from config.settings import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

def allowed_file(filename: str) -> bool:
    """Return True if file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_temp_file(file_storage) -> str:
    """
    Save an uploaded file securely into the temporary uploads folder.
    Returns the absolute saved path.
    """
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    original_name = secure_filename(file_storage.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    file_storage.save(file_path)
    return file_path


def cleanup_file(file_path: str):
    """Safely remove a temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass
