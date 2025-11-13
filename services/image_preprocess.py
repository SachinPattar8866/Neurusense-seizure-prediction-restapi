import os
import json
import numpy as np
from PIL import Image
from utils.errors import APIError
from config.settings import MODEL_METADATA_PATH
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import json


def load_preprocessing_config(model_name: str = "hybrid_best.h5") -> dict:
    """
    Load preprocessing parameters (img_size, color_mode, rescale_factor)
    from backend/models/model_metadata.json.
    """
    if not os.path.exists(MODEL_METADATA_PATH):
        raise APIError(f"Model metadata not found at: {MODEL_METADATA_PATH}", status_code=500)

    try:
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)

        model_entry = metadata.get("models", {}).get(model_name)
        if not model_entry:
            raise APIError(f"No preprocessing config found for '{model_name}'", status_code=500)

        preprocessing = model_entry.get("preprocessing", {})
        return {
            "img_size": (
                int(preprocessing.get("img_size", 128)),
                int(preprocessing.get("img_size", 128)),
            ),
            "color_mode": preprocessing.get("color_mode", "rgb").lower(),
            "rescale_factor": float(preprocessing.get("rescale_factor", 1.0 / 255.0)),
        }

    except Exception as e:
        raise APIError(f"Failed to load preprocessing metadata: {str(e)}", status_code=500)


def preprocess_image(file_storage, model_name: str = "hybrid_best.h5") -> np.ndarray:
    """
    Convert uploaded EEG/spectrogram image into a NumPy array ready for model inference.

    Steps:
      1. Load preprocessing config from model_metadata.json
      2. Convert to color mode (RGB or grayscale)
      3. Resize to expected input size
      4. Normalize using rescale_factor
      5. Add batch dimension → shape (1, H, W, C)
    """
    try:
        config = load_preprocessing_config(model_name)
        img_size = config["img_size"]
        color_mode = config["color_mode"]
        rescale = config["rescale_factor"]

        # Open image
        image = Image.open(file_storage)

        # Convert based on model color mode
        if color_mode == "grayscale":
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        # Resize
        image = image.resize(img_size)

        # Convert to NumPy array and normalize
        image_array = np.array(image).astype("float32") * rescale

        # Add dimensions
        if color_mode == "grayscale" and image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)  # (H, W, 1)
        image_array = np.expand_dims(image_array, axis=0)        # (1, H, W, C)

        return image_array

    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Image preprocessing failed: {str(e)}", status_code=400)


def waveform_image_to_spectrogram_bytes(image_bytes: bytes, model_name: str = "hybrid_best.h5") -> bytes:
    """
    Convert a raw EEG waveform photo (image of a time-series trace) into a
    spectrogram PNG (bytes) matching the training pipeline.

    Approach:
      - Load the uploaded image as grayscale
      - For each column, find the darkest pixel (assumed trace) and convert
        to a 1D signal (height -> amplitude)
      - Smooth and normalize the signal
      - Compute spectrogram using scipy.signal.spectrogram with the same
        parameters as training (nperseg, noverlap, sampling_rate)
      - Render to a PNG using matplotlib with the 'viridis' colormap
    """
    try:
        # Load metadata for spectrogram params
        with open(MODEL_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        model_entry = metadata.get('models', {}).get(model_name, {})
        prep = model_entry.get('preprocessing', {})
        spec_params = prep.get('spectrogram_params', {})
        nperseg = int(spec_params.get('nperseg', 128))
        noverlap = int(spec_params.get('noverlap', 64))
        sfreq = float(prep.get('sampling_rate', 256.0))

        # Open image and convert to grayscale
        img = Image.open(BytesIO(image_bytes)).convert('L')
        w, h = img.size

        # Convert to numpy array (0..255)
        arr = np.array(img).astype('float32')

        # For each column, find weighted darkest pixel (to handle thick traces)
        cols = []
        for x in range(w):
            col = arr[:, x]
            # find indices where intensity is within 10% of min
            minv = np.min(col)
            thresh = minv + 0.1 * (np.max(col) - minv + 1e-8)
            idxs = np.where(col <= thresh)[0]
            if idxs.size == 0:
                # fallback to darkest pixel
                idx = int(np.argmin(col))
            else:
                idx = int(np.mean(idxs))
            # map to amplitude (-1..1)
            amp = (float(h - idx) / float(h)) * 2.0 - 1.0
            cols.append(amp)

        signal = np.array(cols).astype('float32')

        # Simple smoothing to reduce isolated noise
        try:
            from scipy.ndimage import gaussian_filter1d
            signal = gaussian_filter1d(signal, sigma=1.5)
        except Exception:
            pass

        # Ensure there are enough samples; if too short, resample via interpolation
        if signal.size < nperseg:
            # upsample to at least nperseg
            xp = np.linspace(0, 1, signal.size)
            xnew = np.linspace(0, 1, max(nperseg * 2, nperseg))
            signal = np.interp(xnew, xp, signal).astype('float32')

        # Compute spectrogram (magnitude)
        f, t, Sxx = spectrogram(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

        # Normalize similarly to training: dB scale and clip percentiles
        Sxx_db = 10.0 * np.log10(Sxx + 1e-10)
        vmin, vmax = np.percentile(Sxx_db, (5, 95))
        Sxx_db = np.clip(Sxx_db, vmin, vmax)
        Sxx_norm = (Sxx_db - vmin) / (vmax - vmin + 1e-8)

        # Render using matplotlib → PNG bytes (viridis colormap)
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        ax.imshow(Sxx_norm, aspect='auto', origin='lower', cmap='viridis')
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Open as PIL image, convert to RGB, pad/crop to square and resize to model img_size
        try:
            img = Image.open(buf).convert('RGB')

            # pad or crop to square (match training plot_spectrogram_image behavior)
            w, h = img.size
            if w != h:
                side = max(w, h)
                square = Image.new('RGB', (side, side), (0, 0, 0))
                square.paste(img, ((side - w) // 2, (side - h) // 2))
                img = square

            # final resize to model's img_size (if present in metadata)
            img_size = int(prep.get('img_size', 128)) if isinstance(prep.get('img_size', None), (int, str)) else 128
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS
            img = img.resize((img_size, img_size), resample)

            out_buf = BytesIO()
            img.save(out_buf, format='PNG')
            out_buf.seek(0)
            return out_buf.read()
        except Exception:
            # Fallback to raw figure bytes if PIL post-processing fails
            buf.seek(0)
            return buf.read()

    except Exception as e:
        raise APIError(f"Failed to convert waveform image to spectrogram: {e}", status_code=400)
