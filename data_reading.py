import numpy as np

# 加载 .npz
data = np.load("data_rebuild.npz")



# 查看里面都有哪些数组
print("Keys:", data.files)

# 获取 X 和 y
X = data["X"]
y = data["y"]

# 打印它们的形状
print("X.shape =", X.shape)
print("y.shape =", y.shape)

# 查看第一个样本的前几个时间步数据
print("The first sample, the first 20 steps:")
print(X[1700, :10, :, :])  
# print(X[1, :20, 0, 0])

print("Corresponding label:", y[1700])