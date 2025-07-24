import pandas as pd
import matplotlib.pyplot as plt

# 读取前400行扁平化数据
df = pd.read_csv('data_rebuild_flatten.csv', nrows=400)
# df = pd.read_csv('data_rebuild_normalize_flatten.csv', nrows=400)
# df = pd.read_csv('data_rebuild_calib_flatten.csv', nrows=400)
# df = pd.read_csv('data_rebuild_calib_normalize_flatten.csv', nrows=400)


# 提取特征列（最多到 f1300）
features = [c for c in df.columns if c != 'label'][:2595]  # f0...f1300
X = df[features].values  # shape (400, num_features)

# 设置画布：一张图四个子图
fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
colors = ['blue', 'green', 'red', 'purple']
# fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
# colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']

for idx, ax in enumerate(axes.flatten()):
    start = idx * 100
    end = start + 100
    for i in range(start, end):
        ax.plot(range(len(features)), X[i], color=colors[idx], alpha=0.5)
    ax.set_title(f'Samples {start}-{end-1}', fontsize=12)
    ax.grid(True)

# 公共标签和标题
fig.text(0.5, 0.04, 'Feature Index (f0 to f1300)', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=14)
fig.suptitle('Data Flatten Samples Comparison', fontsize=16)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
plt.show()
