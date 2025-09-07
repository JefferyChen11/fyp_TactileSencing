import os
import glob
import numpy as np
import pandas as pd

# === Configuration area ===
ROOT_PAT        = "data/multigrasp_test/*_multigrasp"   # top-level directory pattern (keep consistent with current layout)
CALIB_SUFFIX    = "_merge_calib"                        # only process calibrated merged data
DESIRED_T       = 850                                   # align to 850 frames
OUTPUT_PREFIX   = "data_rebuild_val"                    # output prefix

# -- cropping window: remove first/last 300 frames, keep [300, 550) --
CROP_START = 300
CROP_END   = 550     # exclusive, final length = 250
assert (CROP_END - CROP_START) == 250, "Crop length must be 250 frames"

# -- canonical class order source used to align with training (prefer NPZ 'classes', fallback TXT) --
TRAIN_NPZ_FOR_CLASSES_PRIMARY   = "data_rebuild_merge_calib.npz"       # un-cropped npz (produced by data_processing_all)
TRAIN_NPZ_FOR_CLASSES_FALLBACK  = "data_rebuild_merge_calib_crop.npz"  # cropped npz (produced by data_processing_all)
TRAIN_CLASSES_TXT               = "data_rebuild_merge_calib_classes.txt"

NUM_SENSORS = 8
NUM_AXES    = 3
F_EXPECT    = NUM_SENSORS * NUM_AXES  # 24
EPS = 1e-8

def load_canonical_classes():
    """
    Load the canonical class order used during training, so validation label encoding
    matches training exactly. Prefer the 'classes' field from training NPZ; if missing,
    read from *_classes.txt.
    """
    for npz_path in [TRAIN_NPZ_FOR_CLASSES_PRIMARY, TRAIN_NPZ_FOR_CLASSES_FALLBACK]:
        if os.path.exists(npz_path):
            try:
                d = np.load(npz_path, allow_pickle=True)
                if 'classes' in d.files:
                    cls = [str(x) for x in d['classes'].tolist()]
                    if len(cls) > 0:
                        print(f"[INFO] Loaded {len(cls)} classes from {npz_path}.")
                        return cls
            except Exception as e:
                print(f"[WARN] Failed to read classes from {npz_path}: {e}")
    if os.path.exists(TRAIN_CLASSES_TXT):
        with open(TRAIN_CLASSES_TXT, "r", encoding="utf-8") as f:
            cls = [ln.strip() for ln in f if ln.strip()]
        if len(cls) > 0:
            print(f"[INFO] Loaded {len(cls)} classes from {TRAIN_CLASSES_TXT}.")
            return cls
    raise RuntimeError(
        "Could not find training classes: please run data_processing_all.py to generate a NPZ with 'classes' or create *_classes.txt."
    )

def gather_merged_tensors_calib():
    """
    Scan ROOT_PAT/*{CALIB_SUFFIX} for CSV files,
    and encode labels according to the training canonical classes order.
    Returns X: (N, DESIRED_T, 8, 3), y: (N,), classes: List[str]
    """
    canonical = load_canonical_classes()         # canonical order used during training
    cls_map   = {c: i for i, c in enumerate(canonical)}

    arr_list = []
    labels   = []
    per_class_count = {c: 0 for c in canonical}

    all_dirs = sorted(glob.glob(ROOT_PAT))
    if not all_dirs:
        raise RuntimeError(f"No root directories matched: {ROOT_PAT}")

    for root in all_dirs:
        # Scan in canonical order to ensure consistent label encoding
        for cls in canonical:
            folder = os.path.join(root, f"{cls}{CALIB_SUFFIX}")
            if not os.path.isdir(folder):
                continue
            pattern = os.path.join(folder, f"{cls}{CALIB_SUFFIX}_*.csv")
            files = sorted(glob.glob(pattern))
            for file in files:
                df = pd.read_csv(file)
                # safely drop timestamp column if present
                if 'timestamp' in df.columns:
                    raw = df.drop(columns=['timestamp']).values
                else:
                    raw = df.values

                # align to DESIRED_T: truncate if longer, pad with zeros if shorter
                Tcur, F = raw.shape  # expected F=24
                if F != F_EXPECT:
                    raise ValueError(f"{file} has {F} columns; expected {F_EXPECT} ({NUM_SENSORS} sensors × {NUM_AXES} axes).")
                if Tcur >= DESIRED_T:
                    arr = raw[:DESIRED_T]
                else:
                    pad = np.zeros((DESIRED_T - Tcur, F), dtype=raw.dtype)
                    arr = np.vstack([raw, pad])

                # reshape -> (DESIRED_T, 8, 3)
                arr = arr.reshape(DESIRED_T, NUM_SENSORS, NUM_AXES).astype(np.float32)
                arr_list.append(arr)
                labels.append(cls_map[cls])
                per_class_count[cls] += 1

    if not arr_list:
        raise RuntimeError(f"No calibrated merged CSVs found under pattern: {ROOT_PAT}")

    # print per-class counts
    print("\n[Validation sample counts by class]")
    for cls in canonical:
        print(f"  {cls:20s} -> {per_class_count[cls]}")

    X = np.stack(arr_list)  # (N, DESIRED_T, 8, 3)
    y = np.array(labels, dtype=int)
    print(f"\n[Summary] Validation tensor X.shape={X.shape}, y.shape={y.shape} (T={DESIRED_T})")
    return X, y, canonical

def flatten_sensor_first(X):
    """
    Flatten (N, T, 8, 3) to (N, 24*T), sensor-major:
    for each sensor, concatenate X axis values across time as X(0..T-1), Y(0..T-1), Z(0..T-1),
    then move to next sensor.
    """
    N, T, C, D = X.shape
    return X.transpose(0, 2, 3, 1).reshape(N, C * D * T)

def minmax_norm_along_time(X):
    """
    Perform min-max normalization along time for (N, T, 8, 3).
    Normalization is per-sample, per-sensor, per-axis along the time axis.
    """
    Xmin = X.min(axis=1, keepdims=True)  # (N,1,8,3)
    Xmax = X.max(axis=1, keepdims=True)  # (N,1,8,3)
    return (X - Xmin) / (Xmax - Xmin + EPS)

def save_npz(path, X, y, classes=None):
    if classes is not None:
        np.savez(path, X=X, y=y, classes=np.array(classes, dtype='U'))
    else:
        np.savez(path, X=X, y=y)
    print(f"[SAVED] {path}  shape={X.shape}, labels={y.shape}, classes={len(classes) if classes is not None else 'N/A'}")

def save_flat_csv(path, X_flat, y):
    cols = [f"f{i}" for i in range(X_flat.shape[1])]
    df = pd.DataFrame(X_flat, columns=cols)
    df["label"] = y
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}  shape={(X_flat.shape[0], X_flat.shape[1] + 1)}")

def main():
    # === 1) Read only *_merge_calib, obtain tensors aligned to T=850 and labels encoded by training classes ===
    X, y, classes = gather_merged_tensors_calib()  # (N, 850, 8, 3)
    path_npz_full = f"{OUTPUT_PREFIX}{CALIB_SUFFIX}.npz"
    save_npz(path_npz_full, X, y, classes=classes)

    # optionally save classes.txt for manual inspection
    classes_txt = f"{OUTPUT_PREFIX}{CALIB_SUFFIX}_classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"[SAVED] class list: {classes_txt}  total {len(classes)} classes")

    # === 2) Produce non-cropped flattened CSV and normalized flattened CSV ===
    X_flat = flatten_sensor_first(X)  # (N, 24*850) = (N, 20400)
    save_flat_csv(f"{OUTPUT_PREFIX}{CALIB_SUFFIX}_flatten.csv", X_flat, y)

    X_norm = minmax_norm_along_time(X)  # (N, 850, 8, 3)
    X_flat_norm = flatten_sensor_first(X_norm)  # (N, 20400)
    save_flat_csv(f"{OUTPUT_PREFIX}{CALIB_SUFFIX}_normalize_flatten.csv", X_flat_norm, y)

    # === 3) Crop: keep [300, 550) total 250 frames -> (N, 250, 8, 3) ===
    if X.shape[1] < CROP_END:
        raise ValueError(f"Current sequence length {X.shape[1]} < required crop end {CROP_END}.")
    X_crop = X[:, CROP_START:CROP_END, :, :]  # (N, 250, 8, 3)
    path_npz_crop = f"{OUTPUT_PREFIX}{CALIB_SUFFIX}_crop.npz"
    save_npz(path_npz_crop, X_crop, y, classes=classes)

    # === 4) Produce cropped flattened CSVs and normalized cropped flattened CSVs ===
    Xc_flat = flatten_sensor_first(X_crop)  # (N, 24*250) = (N, 6000)
    save_flat_csv(f"{OUTPUT_PREFIX}{CALIB_SUFFIX}_flatten_crop.csv", Xc_flat, y)

    Xc_norm = minmax_norm_along_time(X_crop)
    Xc_flat_norm = flatten_sensor_first(Xc_norm)  # (N, 6000)
    save_flat_csv(f"{OUTPUT_PREFIX}{CALIB_SUFFIX}_normalize_flatten_crop.csv", Xc_flat_norm, y)

    # === 5) Overview printout ===
    N = X.shape[0]
    print("\n=== Overview ===")
    print(f"N = {N} (if all data present this may be 1500)")
    print(f"Uncropped tensor: {X.shape} -> flattened columns (no label) = {X_flat.shape[1]}  expected 20400")
    print(f"Cropped tensor:   {X_crop.shape} -> flattened columns (no label) = {Xc_flat.shape[1]} expected 6000")

if __name__ == "__main__":
    main()
