import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# ===== defaults =====
ROOTS_DEFAULT = ["data/*_multigrasp"]
ROOTS_200     = ["data/singlegrasp_200/*_200"]
DEFAULT_CALIB_SUFFIX = "_merge_calib"
DEFAULT_DESIRED_T = 850
DEFAULT_CROP_START = 300
DEFAULT_CROP_END_EXCL = 550
DEFAULT_NORMALIZE = "per_sample"
OUTPUT_PREFIX = "data_rebuild"
NUM_SENSORS = 8
NUM_AXES = 3
NUM_FEATURES = NUM_SENSORS * NUM_AXES
EPS = 1e-8
# ====================

def _list_all_roots(root_pats):
    all_dirs = []
    for pat in root_pats:
        all_dirs.extend(sorted(glob.glob(pat)))
    return sorted(all_dirs)

def _discover_class_map(all_dirs, suffix):
    found = set()
    for root in all_dirs:
        if not os.path.isdir(root):
            continue
        for sub in sorted(os.listdir(root)):
            if sub.endswith(suffix):
                cls = sub.replace(suffix, "")
                found.add(cls)
    cls_list = sorted(found)
    return {cls: i for i, cls in enumerate(cls_list)}

def _read_one_csv(file):
    df = pd.read_csv(file, low_memory=False)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if 'timestamp' in num_df.columns:
        num_df = num_df.drop(columns=['timestamp'])
    if num_df.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"[Column count error] {file}\n"
            f"Expected {NUM_FEATURES} columns (= {NUM_SENSORS}×{NUM_AXES}), got {num_df.shape[1]}.\n"
            f"Column names: {list(num_df.columns)}"
        )
    arr = num_df.values.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"[Data error] {file} contains NaN/Inf.")
    return arr

def _flatten_sensor_first(X):
    N, T, C, D = X.shape
    return X.transpose(0, 2, 3, 1).reshape(N, C * D * T)

def _normalize(X, mode="per_sample"):
    if mode == "none":
        return X
    if mode == "per_sample":
        # Min–Max per sample, per sensor, per axis along time
        Xmin = X.min(axis=1, keepdims=True)  # (N,1,C,D)
        Xmax = X.max(axis=1, keepdims=True)  # (N,1,C,D)
        denom = (Xmax - Xmin)
        denom[denom == 0] = 1.0
        return (X - Xmin) / (denom + EPS)
    if mode == "per_channel":
        # Min–Max per channel across the dataset (channels = sensor×axis)
        Xmin = X.min(axis=(0,1), keepdims=True)
        Xmax = X.max(axis=(0,1), keepdims=True)
        denom = (Xmax - Xmin)
        denom[denom == 0] = 1.0
        return (X - Xmin) / (denom + EPS)
    if mode == "global":
        # Global min–max normalization across all values
        Xmin = X.min()
        Xmax = X.max()
        denom = (Xmax - Xmin)
        if denom == 0:
            return np.zeros_like(X)
        return (X - Xmin) / (denom + EPS)
    raise ValueError(f"Unknown NORMALIZE_MODE: {mode}")

def _process_fixed_length(root_pats, suffix, desired_t, stop_on_error=False):
    all_dirs = _list_all_roots(root_pats)
    if not all_dirs:
        raise RuntimeError(f"No root directories matched: {root_pats}")

    cls_map = _discover_class_map(all_dirs, suffix)
    classes = sorted(cls_map.keys(), key=lambda k: cls_map[k])
    print(f"[Class mapping] {cls_map}")

    arr_list, labels = [], []
    per_root = defaultdict(int)
    per_class = defaultdict(int)
    errors = []

    for root in all_dirs:
        if not os.path.isdir(root):
            continue
        for cls, label in cls_map.items():
            folder = os.path.join(root, f"{cls}{suffix}")
            if not os.path.isdir(folder):
                continue
            pattern = os.path.join(folder, f"{cls}{suffix}_*.csv")
            files = sorted(glob.glob(pattern))
            for file in files:
                try:
                    raw = _read_one_csv(file)
                    if raw.shape[0] >= desired_t:
                        cur = raw[:desired_t]
                    else:
                        pad = np.zeros((desired_t - raw.shape[0], raw.shape[1]), dtype=raw.dtype)
                        cur = np.vstack([raw, pad])
                    arr_list.append(cur)
                    labels.append(label)
                    per_root[root] += 1
                    per_class[cls] += 1
                except Exception as e:
                    errors.append(f"{file} -> {e}")
                    if stop_on_error:
                        raise

    if not arr_list:
        raise RuntimeError(
            f"No valid merged data found (suffix {suffix}). Error examples (up to 10):\n" +
            "\n".join(errors[:10])
        )

    # Build X as (N, T, sensors, axes)
    X = np.stack([a.reshape(desired_t, NUM_SENSORS, NUM_AXES) for a in arr_list], dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print("\n[Sample counts - by root]")
    for k in sorted(per_root):
        print(f"  {k}: {per_root[k]}")
    print("\n[Sample counts - by class]")
    for cls in classes:
        print(f"  {cls} (label={cls_map[cls]}): {per_class.get(cls, 0)}")
    print(f"\n[Summary] fixed_length output X.shape={X.shape}, y.shape={y.shape} (DESIRED_T={desired_t})")
    if errors:
        print(f"\n[Warning] Number of files skipped due to errors: {len(errors)} (examples):\n  {errors[:5]}")
    # -- Key: return classes -- #
    return X, y, classes

def save_untrimmed_and_csv(root_pats, suffix, desired_t, normalize_mode, output_prefix, tag="", force=False):
    """
    tag: e.g. "" or "_200" -> used to generate filenames with suffixes
    """
    # -- Obtain X, y, classes -- #
    X_base, y_base, classes = _process_fixed_length(root_pats, suffix, desired_t)

    npz_file = f"{output_prefix}{suffix}{tag}.npz"
    if os.path.exists(npz_file) and not force:
        raise FileExistsError(f"{npz_file} already exists. Use --force to overwrite.")

    # -- Important: save classes in npz as well -- #
    np.savez(npz_file, X=X_base, y=y_base, classes=np.array(classes, dtype='U'))
    print(f"\n[Saved] Untrimmed npz: {npz_file}  shape={X_base.shape}, y.shape={y_base.shape}")

    # Also export a classes.txt (for reuse by validation scripts)
    classes_txt = f"{output_prefix}{suffix}_classes{tag}.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"[Saved] Class list: {classes_txt}  total {len(classes)} classes")

    # Flattened CSV
    flat = _flatten_sensor_first(X_base)
    flat_csv = f"{output_prefix}{suffix}_flatten{tag}.csv"
    df_flat = pd.DataFrame(flat, columns=[f"f{i}" for i in range(flat.shape[1])])
    df_flat['label'] = y_base
    df_flat.to_csv(flat_csv, index=False)
    print(f"[Saved] Untrimmed flattened CSV: {flat_csv}  shape={df_flat.shape}")

    # Normalized -> flattened CSV
    Xn = _normalize(X_base, mode=normalize_mode)
    flatn = _flatten_sensor_first(Xn)
    norm_csv = f"{output_prefix}{suffix}_normalize_flatten{tag}.csv"
    df_flatn = pd.DataFrame(flatn, columns=[f"f{i}" for i in range(flatn.shape[1])])
    df_flatn['label'] = y_base
    df_flatn.to_csv(norm_csv, index=False)
    print(f"[Saved] Untrimmed normalized flattened CSV: {norm_csv}  shape={df_flatn.shape}, mode={normalize_mode}")

    return npz_file

def crop_npz_and_export(npz_path, crop_start, crop_end_excl, normalize_mode, output_prefix, suffix, tag="", force=False):
    data = np.load(npz_path, allow_pickle=True)
    if 'X' not in data.files:
        raise RuntimeError(f"{npz_path} does not contain key 'X', contains: {data.files}")

    X = data['X']
    y = data['y'] if 'y' in data.files else None
    classes = data['classes'] if 'classes' in data.files else None  # -- read classes if present

    if X.ndim != 4:
        raise ValueError(f"Expected X to be 4D (N,T,8,3), but actual shape={X.shape}")

    N, T, C, D = X.shape
    if not (0 <= crop_start < crop_end_excl <= T):
        raise ValueError(f"Crop indices out of range: T={T}, crop_start={crop_start}, crop_end_excl={crop_end_excl}")

    X_crop = X[:, crop_start:crop_end_excl, :, :].astype(np.float32)

    # Output filenames: put '_crop' in the middle, tag at the end
    out_npz = f"{output_prefix}{suffix}_crop{tag}.npz"
    if os.path.exists(out_npz) and not force:
        raise FileExistsError(f"{out_npz} already exists. Use --force to overwrite.")
    # -- Important: save classes into cropped npz as well -- #
    if y is not None and classes is not None:
        np.savez(out_npz, X=X_crop, y=y, classes=classes)
    elif y is not None:
        np.savez(out_npz, X=X_crop, y=y)
    else:
        np.savez(out_npz, X=X_crop)
    print(f"\n[Saved] Cropped npz: {out_npz}  shape={X_crop.shape}")

    # Flattened CSV (cropped)
    flat = _flatten_sensor_first(X_crop)
    flat_csv = f"{output_prefix}{suffix}_crop_flatten{tag}.csv"
    df_flat = pd.DataFrame(flat, columns=[f"f{i}" for i in range(flat.shape[1])])
    if y is not None:
        df_flat['label'] = y
    df_flat.to_csv(flat_csv, index=False)
    print(f"[Saved] Cropped flattened CSV: {flat_csv}  shape={df_flat.shape}")

    # Normalized -> flattened CSV (cropped)
    Xn = _normalize(X_crop, mode=normalize_mode)
    flatn = _flatten_sensor_first(Xn)
    norm_csv = f"{output_prefix}{suffix}_crop_normalize_flatten{tag}.csv"
    df_flatn = pd.DataFrame(flatn, columns=[f"f{i}" for i in range(flatn.shape[1])])
    if y is not None:
        df_flatn['label'] = y
    df_flatn.to_csv(norm_csv, index=False)
    print(f"[Saved] Cropped normalized flattened CSV: {norm_csv}  shape={df_flatn.shape}, mode={normalize_mode}")

def run_for_group(root_pats, suffix, desired_t, crop_start, crop_end_excl, normalize_mode, output_prefix, tag="", force=False, stop_on_error=False):
    """
    Run a full pipeline for one group of roots: fixed_length -> save -> crop
    tag is used for filename tagging (e.g. '' or '_200')
    """
    print(f"\n=== Starting group processing tag='{tag}' roots={root_pats} ===")
    npz_main = save_untrimmed_and_csv(root_pats, suffix, desired_t, normalize_mode, output_prefix, tag=tag, force=force)
    crop_npz_and_export(npz_main, crop_start, crop_end_excl, normalize_mode, output_prefix, suffix, tag=tag, force=force)

def main():
    p = argparse.ArgumentParser(
        description="Generate two groups (default + _200) of untrimmed and cropped data (and their flattened/normalized CSVs)."
    )
    p.add_argument("--desired-t", type=int, default=DEFAULT_DESIRED_T)
    p.add_argument("--crop-start", type=int, default=DEFAULT_CROP_START)
    p.add_argument("--crop-end", type=int, default=DEFAULT_CROP_END_EXCL)
    p.add_argument("--normalize", choices=["per_sample","per_channel","global","none"], default=DEFAULT_NORMALIZE)
    p.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    p.add_argument("--suffix", default=DEFAULT_CALIB_SUFFIX)
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--stop-on-error", action="store_true", help="Stop on first file error (otherwise skip errored files)")
    args = p.parse_args()

    # First group (original)
    run_for_group(
        root_pats = ROOTS_DEFAULT,
        suffix = args.suffix,
        desired_t = args.desired_t,
        crop_start = args.crop_start,
        crop_end_excl = args.crop_end,
        normalize_mode = args.normalize,
        output_prefix = args.output_prefix,
        tag = "",  # original files have no tag
        force = args.force,
        stop_on_error = args.stop_on_error
    )

    # Second group (_200) - uncomment if needed (currently enabled)
    run_for_group(
        root_pats = ROOTS_200,
        suffix = args.suffix,
        desired_t = args.desired_t,
        crop_start = args.crop_start,
        crop_end_excl = args.crop_end,
        normalize_mode = args.normalize,
        output_prefix = args.output_prefix,
        tag = "_200",
        force = args.force,
        stop_on_error = args.stop_on_error
    )

    print("\n=== All groups processed ===")

if __name__ == "__main__":
    main()
