import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASELINE_CSV = "data_rebuild_merge_calib_crop_flatten_200.csv"
# BASELINE_CSV = "data_rebuild_merge_calib_flatten_200.csv"

N_ROWS = 200
N_COLS = 2550
OUT_PNG = "baseline_first200_2550.png"

def load_first_rows_prefix(csv_path, n_rows, n_cols):
    # read only the first n_rows to keep memory usage small (200 x 2550 is tiny)
    df = pd.read_csv(csv_path, nrows=n_rows)
    # remove the label column if present
    if 'label' in df.columns:
        df_vals = df.drop(columns=['label']).values
    else:
        df_vals = df.values
    df_vals = df_vals.astype(np.float32)
    return df_vals, df.shape[0], df.shape[1]  # values, actual_rows_read, original_num_columns_in_file (incl label if present)

def ensure_cols_and_stack(rows_array, target_cols):
    # rows_array shape: (R, C) where C = original cols (after dropping label)
    R, C = rows_array.shape
    if C >= target_cols:
        stacked = rows_array[:, :target_cols]
    else:
        # pad zeros at tail for each row
        pad_width = target_cols - C
        stacked = np.pad(rows_array, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
    return stacked

def plot_stack(data_matrix, out_png):
    # data_matrix: (n_rows, n_cols)
    n_rows, n_cols = data_matrix.shape
    x = np.arange(n_cols)

    plt.figure(figsize=(16,6))
    # use low alpha for many lines to avoid overplot saturation
    alpha = 0.12 if n_rows >= 100 else 0.2
    lw = 0.9

    for i in range(n_rows):
        plt.plot(x, data_matrix[i], linewidth=lw, alpha=alpha)

    # draw the mean curve (prominent)
    mean_curve = data_matrix.mean(axis=0)
    plt.plot(x, mean_curve, linewidth=2.0, alpha=1.0, label="mean (n={})".format(n_rows), color='black')

    plt.title(f"Baseline _123 — first {n_rows} rows (first {n_cols} columns each)")
    plt.xlabel("Feature index (0 .. {})".format(n_cols-1))
    plt.ylabel("Value")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

def main():
    rows_vals, actual_rows_read, original_cols = load_first_rows_prefix(BASELINE_CSV, N_ROWS, N_COLS)
    print(f"Read {actual_rows_read} rows from CSV (original file columns: {original_cols}).")
    if actual_rows_read < N_ROWS:
        print(f"Warning: CSV has only {actual_rows_read} rows; will plot those.")
    stacked = ensure_cols_and_stack(rows_vals, N_COLS)
    plot_stack(stacked, OUT_PNG)
    print(f"Saved plot to {OUT_PNG}.")

if __name__ == "__main__":
    main()