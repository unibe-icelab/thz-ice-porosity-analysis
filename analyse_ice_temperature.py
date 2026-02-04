from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile
from math_utils import get_fft

from utils import get_thz_files

if __name__ == '__main__':

    path = Path("/Users/linus/Documents/solid_ice_5mm_trans_temperature_profile/data_image")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    for file in get_thz_files(path):
        with DotthzFile(file) as f:
            print(file)
            t = f["Image"].datasets["time"][:]
            d = f["Image"].datasets["dataset"][:]
            d = np.mean(d, axis=(0, 1))

            d = d[t < 1960]
            t = t[t < 1960]

            axes[0].plot(t, d)

            f, a, arg = get_fft(t, d)

            axes[1].plot(f, a)

    plt.show()
