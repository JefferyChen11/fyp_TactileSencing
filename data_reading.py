import numpy as np

# load .npz
data = np.load("data_rebuild_merge.npz")
# data = np.load("data_rebuild_merge_calib.npz")
# data = np.load("data_rebuild_val_merge.npz")
# data = np.load("data_rebuild_val_merge_calib.npz")


# show which arrays are inside
print("Keys:", data.files)

# get X and y
X = data["X"]
y = data["y"]

# print their shapes
print("X.shape =", X.shape)
print("y.shape =", y.shape)

# show the first few time steps of the first sample
print("The first sample, first 5 time steps:")
print(X[0, :5, :, :])
# print(X[1, :20, 0, 0])

print("Corresponding label:", y[0])
