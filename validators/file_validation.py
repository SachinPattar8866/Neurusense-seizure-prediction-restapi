import os
from werkzeug.utils import secure_filename
from utils.errors import APIError
from config.settings import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH

def allowed_file(filename: str) -> bool:
    """
    Check if the file has an allowed extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file_storage):
    """
    Validate uploaded file:
    - Not empty
    - Allowed extension (png/jpg/jpeg)
    - Within size limit (based on MAX_CONTENT_LENGTH)
    """
    if not file_storage or file_storage.filename == "":
        raise APIError("No file uploaded", status_code=400)

    filename = secure_filename(file_storage.filename)

    if not allowed_file(filename):
        raise APIError("Invalid file type. Only PNG, JPG, JPEG allowed.", status_code=400)

    # Check file size
    file_storage.seek(0, os.SEEK_END)
    file_size_bytes = file_storage.tell()
    file_storage.seek(0)

    max_bytes = MAX_CONTENT_LENGTH
    if file_size_bytes > max_bytes:
        max_mb = round(max_bytes / (1024 * 1024), 2)
        raise APIError(f"File too large. Max {max_mb} MB allowed.", status_code=400)

    return filename
