
---

# 📄 `docs/model_card.md`

```markdown
# Model Card – Neurosense Seizure Prediction System

## Overview
The Neurosense system uses deep learning to predict epileptic seizures from EEG image data.  
It has two models:
- **Hybrid CNN + BiLSTM** → Final deployed model (production)
- **CNN baseline** → Benchmark model for comparison

---

## Dataset
- **Source:** CHB-MIT Scalp EEG Dataset
- **Input Data:** EEG signals converted into spectrogram images (0.5–70 Hz filtered, normalized, windowed)
- **Preprocessing:** 
  - Artifact removal
  - Normalization per channel
  - Spectrogram conversion for CNN input

---

## Models
### 1. CNN Baseline
- Purpose: Benchmark
- Architecture: Convolutional layers + dense classifier
- Size: ~5 MB
- Stored at: `models/cnn_baseline.h5`

### 2. Hybrid CNN + BiLSTM
- Purpose: Production deployment
- Architecture: CNN feature extractor + BiLSTM temporal learning
- Size: ~20 MB
- Stored at: `models/hybrid_cnn_bilstm.h5`

---

## Performance (Example – replace with your own results)
- **CNN Baseline:**  
  - Accuracy: 84%  
  - Sensitivity: 78%  
  - Specificity: 86%  
  - F1-Score: 0.80  

- **Hybrid CNN+BiLSTM:**  
  - Accuracy: 92%  
  - Sensitivity: 90%  
  - Specificity: 93%  
  - F1-Score: 0.91  

---

## Limitations
- Trained only on CHB-MIT pediatric dataset → may not generalize to adults.
- Works only with EEG image spectrograms (not raw EDF files directly).
- Not a replacement for a medical diagnosis.

---

## Ethical Considerations
- Designed as a **decision-support tool**, not an autonomous medical device.
- Alerts should always be reviewed by a clinician.
- Intended for hospital and research environments.
