import os
import glob
import numpy as np
import pandas as pd

# === 配置区域 ===
ROOT_PAT        = "data/*_100"     # 一级目录模式
MERGE_SUFFIX    = ""         # 原始数据合并后文件夹后缀
CALIB_SUFFIX    = "_calib"   # 校准数据合并后文件夹后缀
TRIM_FRONT      = 0               # 修剪前丢弃帧数
TRIM_BACK       = 0               # 修剪后丢弃帧数
OUTPUT_PREFIX   = "data_rebuild"   # 输出文件名前缀

def gather_merged_tensors(suffix):
    """
    扫描 data/*_100/*{suffix} 文件夹，
    返回 tensors (N, T_trimmed, 8, 3) 和 labels (N,)
    """
    arr_list = []
    labels   = []

    # 1) 自动发现所有类别并构建标签映射
    all_dirs = sorted(glob.glob(ROOT_PAT))
    cls_map  = {}
    next_label = 0
    for root in all_dirs:
        for sub in sorted(os.listdir(root)):
            if sub.endswith(suffix):
                cls = sub.replace(suffix, "")
                if cls not in cls_map:
                    cls_map[cls] = next_label
                    next_label += 1

    # 2) 读取各类别下的所有合并CSV
    for root in all_dirs:
        for cls, label in cls_map.items():
            merge_folder = os.path.join(root, f"{cls}{suffix}")
            if not os.path.isdir(merge_folder):
                continue
            pattern = os.path.join(merge_folder, f"{cls}{suffix}_*.csv")
            for file in glob.glob(pattern):
                df = pd.read_csv(file)
                raw = df.drop(columns=['timestamp']).values
                arr_list.append(raw)
                labels.append(label)

    if not arr_list:
        raise RuntimeError(f"未找到任何后缀为 '{suffix}' 的合并数据，请先运行合并脚本。")

    # 3) 对齐到最短时间步并裁剪
    min_T = min(a.shape[0] for a in arr_list)
    aligned = []
    for arr in arr_list:
        # pad / 截断到 min_T
        if arr.shape[0] > min_T:
            arr2 = arr[:min_T]
        else:
            pad = np.zeros((min_T - arr.shape[0], arr.shape[1]))
            arr2 = np.vstack([arr, pad]) if arr.shape[0] < min_T else arr

        # 裁剪前后帧
        if arr2.shape[0] > TRIM_FRONT + TRIM_BACK:
            arr2 = arr2[TRIM_FRONT:arr2.shape[0]-TRIM_BACK]
        aligned.append(arr2)

    # 4) reshape 成 (N, T, 8, 3)
    final_T = aligned[0].shape[0]
    tensors = np.stack([a.reshape(final_T, 8, 3) for a in aligned])
    return tensors, np.array(labels)

def save_rebuild_and_flatten(suffix):
    """
    对给定 suffix（_merge 或 _merge_calib），
    生成 1) 重塑 npz，2) 扁平化 csv，3) 归一化扁平化 csv
    """
    X, y = gather_merged_tensors(suffix)

    # 1) 保存重塑数据
    npz_file = f"{OUTPUT_PREFIX}{suffix}.npz"
    np.savez(npz_file, X=X, y=y)
    print(f"保存重塑数据: {npz_file}, shape={X.shape}")

    # 2) 扁平化并保存
    N, T, C, D = X.shape
    flat = X.reshape(N, C*D*T)
    cols = [f"f{i}" for i in range(flat.shape[1])]
    df_flat = pd.DataFrame(flat, columns=cols)
    df_flat['label'] = y
    flat_csv = f"{OUTPUT_PREFIX}{suffix}_flatten.csv"
    df_flat.to_csv(flat_csv, index=False)
    print(f"保存扁平化数据: {flat_csv}, shape={flat.shape}")

    # 3) 归一化扁平化并保存
    eps = 1e-8
    Xmin = X.min(axis=1, keepdims=True)
    Xmax = X.max(axis=1, keepdims=True)
    norm = (X - Xmin) / (Xmax - Xmin + eps)
    flat_norm = norm.reshape(N, C*D*T)
    df_norm = pd.DataFrame(flat_norm, columns=cols)
    df_norm['label'] = y
    norm_csv = f"{OUTPUT_PREFIX}{suffix}_normalize_flatten.csv"
    df_norm.to_csv(norm_csv, index=False)
    print(f"保存归一化扁平化数据: {norm_csv}, shape={flat_norm.shape}")

def main():
    # 先处理原始 merge，再处理 calib
    save_rebuild_and_flatten(MERGE_SUFFIX)
    save_rebuild_and_flatten(CALIB_SUFFIX)

if __name__ == '__main__':
    main()
