import numpy as np
import h5py

# Load dữ liệu từ .npy
X = np.load("X.npy")
y = np.load("y.npy")

# Lưu dữ liệu sang .h5
with h5py.File("data.h5", "w") as f:
    f.create_dataset("X", data=X)
    f.create_dataset("y", data=y)