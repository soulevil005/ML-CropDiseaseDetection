# ML-CropDiseaseDetection
## Crop Disease Detection — Project Description

This project implements an image-based **Crop Disease Detection** pipeline using convolutional neural networks (TensorFlow / Keras). The model classifies plant-leaf images into multiple disease classes (plus healthy) so farmers and agronomists can quickly identify leaf-level disease and get suggested treatments.

**Dataset used:** New Plant Diseases Dataset (Kaggle) — contains ~87k RGB images of healthy and diseased crop leaves, organized by class (≈38 classes). :contentReference[oaicite:1]{index=1}

### Key features
- Simple, extendable CNN classifier built with TensorFlow / Keras
- Data augmentation (rotation, shifts, shear, zoom, flips)
- Training pipeline using `ImageDataGenerator` for efficient image loading
- Class mapping saved as `class_names.npy` for reproducible inference
- Demo inference script which displays predicted class, confidence, and suggested treatment (from `disease_treatments.txt`)
- Model exported as Keras SavedModel / `.keras` file for portability

### Outputs produced
- `models/crop_disease_model.keras` — trained Keras model
- `class_names.npy` — ordered array of class labels (same order as training generator)
- `disease_treatments.txt` — optional human-authored mapping of disease -> treatment
- training plots & logs (optional)

### Quick usage
1. Download dataset (see `dataset_instructions.txt`).
2. Prepare environment: `pip install -r requirements.txt`
3. Train: `python src/train.py --train_dir data/raw/train --valid_dir data/raw/valid --epochs 10`
4. Run demo inference: `python src/test.py --model models/crop_disease_model.keras --image path/to/image.jpg --treatments data/disease_treatments.txt`

### Notes
- Do **not** upload the raw dataset to GitHub (size constraints). Use Kaggle CLI or store dataset externally. :contentReference[oaicite:2]{index=2}
- The code is intentionally simple so you can easily swap the backbone (e.g., use `tf.keras.applications.ResNet50` or EfficientNet for best performance).
