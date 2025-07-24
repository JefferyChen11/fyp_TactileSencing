import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

# === 参数配置 ===
FLATTEN_CSV       = "data_rebuild_normalize_flatten.csv"  # 使用 data_processing_all 生成的归一化扁平化 CSV
MODEL_FILE        = "svm_model.joblib"            # 训练后模型保存路径
NUM_RUNS          = 20                              # 运行次数
BASE_RANDOM_STATE = 42                              # 随机种子起始值

# 划分比例
TRAIN_RATIO = 0.7
TEST_RATIO  = 0.2
# VAL_RATIO 隐含为剩余部分

# 固定类别顺序及对应标签 0-29
CLASS_NAMES = [
    'bigberry_ds20', 'citrus_ds20', 'rough_ds20', 'smallberry_ds20', 'smooth_ds20', 'strawberry_ds20',
    'bigberry_ds30', 'citrus_ds30', 'rough_ds30', 'smallberry_ds30', 'smooth_ds30', 'strawberry_ds30',
    'bigberry_ef10', 'citrus_ef10', 'rough_ef10', 'smallberry_ef10', 'smooth_ef10', 'strawberry_ef10',
    'bigberry_ef30', 'citrus_ef30', 'rough_ef30', 'smallberry_ef30', 'smooth_ef30', 'strawberry_ef30',
    'bigberry_ef50', 'citrus_ef50', 'rough_ef50', 'smallberry_ef50', 'smooth_ef50', 'strawberry_ef50'
]


def load_flatten_csv(path):
    df = pd.read_csv(path)
    y = df['label'].values.astype(int)
    X = df.drop(columns=['label']).values
    return X, y


def split_by_class(X, y, train_ratio, test_ratio, random_state):
    """
    按类别分层随机划分训练/测试/验证集，返回各子集
    """
    idx_by_class = defaultdict(list)
    for idx, label in enumerate(y):
        idx_by_class[label].append(idx)

    train_idx, test_idx, val_idx = [], [], []
    rng = np.random.RandomState(random_state)
    for label, inds in idx_by_class.items():
        arr = np.array(inds)
        rng.shuffle(arr)
        n_total = len(arr)
        n_train = int(n_total * train_ratio)
        n_test  = int(n_total * test_ratio)
        n_val   = n_total - n_train - n_test

        train_idx.extend(arr[:n_train])
        test_idx.extend(arr[n_train:n_train + n_test])
        val_idx.extend(arr[n_train + n_test:])

    return (
        X[train_idx], y[train_idx],
        X[test_idx],  y[test_idx],
        X[val_idx],   y[val_idx]
    )


def main():
    # 载入数据
    X, y = load_flatten_csv(FLATTEN_CSV)
    print(f"检测到 {len(CLASS_NAMES)} 个类别。标签对应: {CLASS_NAMES}")

    summary = {'train': [], 'test': [], 'val': []}
    model = None

    for run in range(NUM_RUNS):
        seed = BASE_RANDOM_STATE + run
        X_train, y_train, X_test, y_test, X_val, y_val = split_by_class(
            X, y, TRAIN_RATIO, TEST_RATIO, seed)

        # 训练并评估
        model = SVC(kernel='rbf', random_state=seed)
        model.fit(X_train, y_train)

        print(f"\n========== Run {run+1} (seed={seed}) ==========")
        for split_name, Xs, ys in [('训练集', X_train, y_train),
                                   ('测试集', X_test, y_test),
                                   ('验证集', X_val, y_val)]:
            y_pred = model.predict(Xs)
            acc = np.mean(y_pred == ys)
            print(f"--- {split_name} ---")
            print(f"准确率: {acc:.4f}")
            print(classification_report(ys, y_pred, target_names=CLASS_NAMES))
            print(confusion_matrix(ys, y_pred))

            key = 'train' if split_name == '训练集' else 'test' if split_name == '测试集' else 'val'
            summary[key].append(acc)

    # 保存最后一次训练的模型
    if model is not None:
        joblib.dump(model, MODEL_FILE)
        print(f"\n模型已保存到: {MODEL_FILE}")

    # 打印汇总结果
    print("\n===== 多次运行 平均 ± 标准差 =====")
    for key in ['train', 'test', 'val']:
        arr = np.array(summary[key])
        print(f"{key}: {arr.mean():.4f} ± {arr.std():.4f}")


if __name__ == '__main__':
    main()