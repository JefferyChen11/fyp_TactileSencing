import os
import pandas as pd

# === Setting ===
DATASETS = [
    ("ecoflex10",    "data/ef10_100", {
        "bigberry_ef10":        0,
        "citrus_ef10":          1,
        "rough_ef10":           2,
        "smallberry_ef10":      3,
        "smooth_ef10":          4,
        "strawberry_ef10":      5
    }),
    ("ecoflex30",    "data/ef30_100", {
        "bigberry_ef30":        0,
        "citrus_ef30":          1,
        "rough_ef30":           2,
        "smallberry_ef30":      3,
        "smooth_ef30":          4,
        "strawberry_ef30":      5
    }),
    ("ecoflex50",    "data/ef50_100", {
        "bigberry_ef50":        0,
        "citrus_ef50":          1,
        "rough_ef50":           2,
        "smallberry_ef50":      3,
        "smooth_ef50":          4,
        "strawberry_ef50":      5
    }),
    ("dragonSkin20",    "data/ds20_100", {
        "bigberry_ds20":        0,
        "citrus_ds20":          1,
        "rough_ds20":           2,
        "smallberry_ds20":      3,
        "smooth_ds20":          4,
        "strawberry_ds20":      5
    }),
    ("dragonSkin30", "data/ds30_100", {
        "bigberry_ds30":        0,
        "citrus_ds30":          1,
        "rough_ds30":           2,
        "smallberry_ds30":      3,
        "smooth_ds30":          4,
        "strawberry_ds30":      5
    })
]
MERGE_TOL_MS = 10  # Merge tolerance in milliseconds


def normalize_ts(s: str) -> str:
    """
    Normalize timestamp string to ensure it has 6 decimal places.
    If no decimal point, append '.000000'.
    """
    s = s.strip()
    if '.' not in s:
        return s + '.000000'
    head, frac = s.split('.', 1)
    frac = ''.join(filter(str.isdigit, frac))
    return f"{head}.{(frac + '000000')[:6]}"


def load_sensor_pair(df0: pd.DataFrame, df1: pd.DataFrame, tol_ms: int = MERGE_TOL_MS) -> pd.DataFrame:
    """
    Merge two DataFrames with raw sensor data, aligning them within a specified tolerance.
    """
    df0 = df0.copy()
    df1 = df1.copy()
    df0['timestamp'] = pd.to_datetime(
        df0['timestamp'].astype(str).map(normalize_ts),
        format="%Y-%m-%dT%H:%M:%S.%f"
    )
    df1['timestamp'] = pd.to_datetime(
        df1['timestamp'].astype(str).map(normalize_ts),
        format="%Y-%m-%dT%H:%M:%S.%f"
    )
    df0.sort_values('timestamp', inplace=True)
    df1.sort_values('timestamp', inplace=True)

    merged = pd.merge_asof(
        df0,
        df1,
        on='timestamp',
        suffixes=('_s0','_s1'),
        direction='nearest',
        tolerance=pd.Timedelta(f"{tol_ms}ms")
    ).dropna()

    # Collect raw axes columns x1–z4 from both sensors
    axes = []
    for suf in ['_s0','_s1']:
        for i in range(1,5):
            for ax in ['x','y','z']:
                axes.append(f"{ax}{i}{suf}")

    return merged[['timestamp'] + axes]


def load_sensor_pair_calib(df0: pd.DataFrame, df1: pd.DataFrame, tol_ms: int = MERGE_TOL_MS) -> pd.DataFrame:
    """
    Merge two DataFrames with calibrated sensor data, aligning them within a specified tolerance
    and rounding to one decimal place.
    """
    df0 = df0.copy()
    df1 = df1.copy()
    df0['timestamp'] = pd.to_datetime(
        df0['timestamp'].astype(str).map(normalize_ts),
        format="%Y-%m-%dT%H:%M:%S.%f"
    )
    df1['timestamp'] = pd.to_datetime(
        df1['timestamp'].astype(str).map(normalize_ts),
        format="%Y-%m-%dT%H:%M:%S.%f"
    )
    df0.sort_values('timestamp', inplace=True)
    df1.sort_values('timestamp', inplace=True)

    merged = pd.merge_asof(
        df0,
        df1,
        on='timestamp',
        suffixes=('_s0','_s1'),
        direction='nearest',
        tolerance=pd.Timedelta(f"{tol_ms}ms")
    ).dropna()

    # Collect calibrated axes columns x1_calib–z4_calib from both sensors
    calib_axes = []
    for suf in ['_s0','_s1']:
        for i in range(1,5):
            for ax in ['x','y','z']:
                calib_axes.append(f"{ax}{i}_calib{suf}")

    df_calib = merged[['timestamp'] + calib_axes].copy()
    # Round calibration values to one decimal place
    df_calib[calib_axes] = df_calib[calib_axes].round(1)
    return df_calib


def merge_all():
    for _, data_dir, cls_map in DATASETS:
        for cls_name in cls_map:
            cls_folder       = os.path.join(data_dir, cls_name)
            merge_folder     = os.path.join(data_dir, f"{cls_name}_merge")
            merge_calib_folder = os.path.join(data_dir, f"{cls_name}_merge_calib")
            os.makedirs(merge_folder, exist_ok=True)
            os.makedirs(merge_calib_folder, exist_ok=True)

            # Scan all sensor0/sensor1 CSV files
            files = [fn for fn in os.listdir(cls_folder) if fn.endswith('.csv')]
            groups = {}
            for fn in files:
                if not fn.startswith('sensor') or '_data_' not in fn:
                    continue
                sensor_id, rest = fn.split('_data_', 1)
                ts = rest[:-4]  # remove .csv
                groups.setdefault(ts, {})[sensor_id] = fn

            # Merge pairs of sensor files for both raw and calibrated data
            raw_count = 0
            calib_count = 0
            for ts, pair in groups.items():
                if 'sensor0' in pair and 'sensor1' in pair:
                    path0 = os.path.join(cls_folder, pair['sensor0'])
                    path1 = os.path.join(cls_folder, pair['sensor1'])
                    df0 = pd.read_csv(path0)
                    df1 = pd.read_csv(path1)

                    # Raw data merge
                    df_raw = load_sensor_pair(df0, df1)
                    if not df_raw.empty:
                        out_raw = os.path.join(merge_folder, f"{cls_name}_merge_{ts}.csv")
                        df_raw.to_csv(out_raw, index=False)
                        raw_count += 1

                    # Calibrated data merge
                    df_calib = load_sensor_pair_calib(df0, df1)
                    if not df_calib.empty:
                        out_calib = os.path.join(merge_calib_folder, f"{cls_name}_merge_calib_{ts}.csv")
                        df_calib.to_csv(out_calib, index=False)
                        calib_count += 1

            print(f"[{cls_name}] Raw merge: {raw_count} files → {merge_folder}")
            print(f"[{cls_name}] Calib merge: {calib_count} files → {merge_calib_folder}")


if __name__ == '__main__':
    merge_all()
