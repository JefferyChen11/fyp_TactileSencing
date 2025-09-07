# Stiffness and Texture Classification with 2D Inputs based on Cilia Tactile Sensor

> Project: time-series tactile classification (stiffness × texture) using data from a cilia-based tactile sensor.
> This repository contains preprocessing pipelines and multiple classification approaches: classical ML (SVM, Bayes Inferen) and hybrid deep-learning + SVM (LSTM encoder + SVM).

code github link: https://github.com/JefferyChen11/fyp_TactileSencing.git
---

## Repository overview

This documentation outlines the purpose and usage of the primary scripts. The code files are parallel; each script focuses on one step of the pipeline.

### Key scripts (brief)
- `merge_sensor_data.py` — Merge sensor0 and sensor1 CSVs by timestamp into a single calibrated CSV per sample (`*_merge_calib.csv`). Use this first to create merged, calibrated data.  
- `data_processing_all.py` — Full preprocessing pipeline for training data. Reads `data/*_multigrasp` and `data/singlegrasp_200/*_200` `*_merge_calib` files and performs rebuild, flatten, normalization, and crop. Outputs NPZ and flattened CSV files used by models.  
- `data_processing_val.py` — Validation set preprocessing. Processes `data/multigrasp_test/*_multigrasp` `*_merge_calib` files with the same pipeline as training; ensures label encoding matches training (reads classes from training NPZ or text).  
- `bayes_model.py` — Train and evaluate Bayes model using `data_rebuild_merge_calib_crop_normalize_flatten.csv`. Performs cross-validation and saves `bayes_model.joblib`.  
- `svm_model.py` — Train and evaluate SVM (GridSearch + StratifiedKFold) on the same flattened CSV and save `svm_model.joblib`.  
- `validation.py` — Use trained `svm` or `bayes` models to validate on `data_rebuild_val_merge_calib_crop_normalize_flatten.csv`; prints classification report and confusion matrix.  
- `lstm_svm_model.py` — LSTM encoder + SVM pipeline. Train an LSTM encoder on sequences, extract features, then train an SVM on encoded features. Saves `lstm_encoder.pt` and `lstm_svm_model.joblib`.  
- `lstm_svm_lessfeature.py` — Variant of the LSTM+SVM pipeline using fewer features / a reduced 4×4 (16-class) subset.  
- `lstm_svm_validation.py` / `lstm_svm_validation_lessfeature.py` — Validation scripts for the LSTM+SVM pipelines. Include TTA (time-warp/shift) and optional CORAL domain alignment before final prediction.

---

## Expected data layout

```
project_root/
├─ data/
│  ├─ <class>_100/                    # training class folders
│  │   └─ <sample>_merge_calib.csv
│  ├─ <class>_multigrasp/             # training class folders
│  │   └─ <sample>_merge_calib.csv
│  ├─ multigrasp_test/                # validation folders
│  │   └─ <sample>_merge_calib.csv
│  └─ singlegrasp_200/                # extra single-grasp samples (_200)
│      └─ <...>_200/
```

`merge_sensor_data.py` reads raw sensor CSVs and produces calibrated merged CSVs under `*_merge_calib` directories.

---

## Typical pipeline (commands)

1. **Merge sensor pairs**
```bash
python merge_sensor_data.py
```
Generates merged and calibrated CSVs (`*_merge_calib.csv`) for each sample.

2. **Preprocess training data**
```bash
python data_processing_all.py --desired-t 850 --crop-start 300 --crop-end 550 --normalize per_sample
```
Outputs (examples):
- `data_rebuild_merge_calib.npz`
- `data_rebuild_merge_calib_crop.npz`
- `data_rebuild_merge_calib_crop_normalize_flatten.csv`  (used by SVM/Bayes)

3. **Preprocess validation data**
```bash
python data_processing_val.py
```
This reads training `classes` to ensure label encoding consistency and outputs `data_rebuild_val_merge_calib_crop.npz` and flattened CSVs.

4. **Train classical models**
- Gaussian Naive Bayes:
```bash
python bayes_model.py
```
- SVM:
```bash
python svm_model.py
```

5. **Validate classical models**
```bash
python validation.py --model svm_model.joblib --input data_rebuild_val_merge_calib_crop_normalize_flatten.csv
```

6. **Train LSTM→SVM hybrid**
```bash
python lstm_svm_model.py
```
Saves encoder and SVM models for later inference/validation.

7. **Validate LSTM→SVM hybrid**
```bash
python lstm_svm_validation.py --encoder lstm_encoder.pt --svm lstm_svm_model.joblib --input data_rebuild_val_merge_calib_crop.npz
```

---

## Preprocessing details & shapes

- Each merged CSV (calibrated) contains sensor channels for sensor0 and sensor1. Expected data channels: `N_SENSORS × N_AXES` (e.g., 8 sensors × 3 axes = 24 channels). Timestamps are parsed and then dropped for modeling.
- `DESIRED_T` default: **850** samples. Sequences longer than `DESIRED_T` are truncated; shorter sequences are zero-padded. After cropping (`crop_start=300`, `crop_end=550` by default) sequences of length 250 are typically used for training/encoder input.
- Flattening order: sensor-major → axis → time (final flattened length for cropped sequences = channels × T_crop, e.g., 24 × 250 = 6000).
- Normalization modes supported: `per_sample` (default), `per_channel`, `global`, `none`.

---

## Modeling notes

- **Classical models (Bayes, SVM)** use flattened, normalized CSV as input. Cross-validation (e.g., 5-fold) runs during training scripts and results (confusion matrices, classification reports) are saved/printed.
- **LSTM encoder** trains on time-series (shape: samples × channels × T) and produces a fixed-length embedding per sample. The downstream SVM is trained on these embeddings.
- **TTA (test-time augmentation)** for LSTM validation: time scaling and shifts (e.g., scales `[0.95,1.0,1.05]`, shifts `[-60,-30,0,30,60]`) are used to produce multiple predictions which are averaged.
- **CORAL domain alignment** (optional in validation scripts) aligns validation features to the training feature distribution before SVM classification to reduce domain shift between training and validation sets.
- LSTM scripts use PyTorch; GPU is recommended for faster training. CPU will work but slower.

---

## Common output files

- Preprocessing: `data_rebuild_merge_calib.npz`, `*_crop.npz`, `*_crop_normalize_flatten.csv`.
- Models: `bayes_model.joblib`, `svm_model.joblib`, `lstm_encoder.pt`, `lstm_svm_model.joblib`.
- Visual outputs: confusion matrices, per-class accuracy heatmaps, and any saved OOF predictions.

---

## Dependencies

Minimum tested environment:
- Python 3.8+
- numpy, pandas
- scikit-learn, joblib, matplotlib
- torch (PyTorch) — for LSTM models

Install basics with pip:
```bash
pip install numpy pandas scikit-learn joblib matplotlib torch
```

---

## Debugging tips & gotchas

- `data_processing_val.py` requires the training `classes` mapping (from training NPZ or a `classes.txt`). If missing, label encodings will be inconsistent — the script will raise an informative error.
- Check that merged CSVs contain the expected number of sensor columns (e.g., 24) before running heavy preprocess or model training.
- When training LSTM on CPU, reduce `batch_size` and `epochs` to shorten runtime.
- Always keep a copy of `feature_names` (if produced) when saving SVM models so downstream validation can map features correctly.

---

## Reproducibility & example end-to-end script

Below is a minimal example bash sequence to run the full pipeline (adjust paths/options to your environment):
```bash
# 1. Merge sensor pairs
python merge_sensor_data.py

# 2. Preprocess training data
python data_processing_all.py --desired-t 850 --crop-start 300 --crop-end 550 --normalize per_sample

# 3. Preprocess validation
python data_processing_val.py

# 4. Train SVM
python svm_model.py

# 5. Validate SVM
python validation.py --model svm_model.joblib
```

---

## Next steps (suggestions)
- Add a top-level `run_pipeline.sh` or `Makefile` to reproduce the full experiment with one command.
- Add unit tests for preprocess functions (edge cases: short sequences, missing channels).
- Add Dockerfile + resource-limited runner for safe model evaluation of new/untrusted samples.
- Add more logging and save model training hyperparameters alongside model artifacts.

---

## Contact
Author / maintainer: Your Name — your.email@example.com

If you want any changes to wording, extra sections (e.g., full CLI options for each script), or a Chinese version, tell me and I will update the saved README.
