import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.random_memory import CustomRandomMemory
import numpy as np


if __name__ == "__main__":
    file_dir = "logs/pendulum/memories/"
    # get the latest file
    files = os.listdir(file_dir)
    files = [f for f in files if f.endswith(".npz")]
    file_path = os.path.join(file_dir, sorted(files)[-1])

    data = np.load(file_path, allow_pickle=True)
    print("DEBUG", data.files)
    for key in data.files:
        print("DEBUG", key, data[key].shape)

    print(data["actions"])
