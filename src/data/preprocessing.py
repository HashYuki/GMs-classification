import os
import numpy as np
import glob

from . import utils


def make_data(path, if_save=True, verbose=True):
    # TVC
    if verbose:
        print("preprocessing...")

    # get subject data path as list
    wms_path_list = sorted(glob.glob(f"{path}/WMs/*/*"))
    pr_path_list = sorted(glob.glob(f"{path}/PR/*/*"))

    # marge wms and pr path list
    sub_path_list = wms_path_list + pr_path_list

    # make labels. (wms=0; pr=1)
    sub_labels = [0] * len(wms_path_list) + [1] * len(pr_path_list)

    X = []
    y = []

    for path, label in zip(sub_path_list, sub_labels):
        # get kpts data
        joints = np.load(f"{path}/kpts.npy")

        # utils
        joints = utils.rolling_median(joints, 5)
        joints = utils.rolling_mean(joints, 5)

        X.append(joints[:1800])
        y.append(label)
    X = np.stack(X)  # NTVC
    X = X.transpose(0, 3, 1, 2)  # NCTV
    y = np.array(y)

    if if_save:
        os.makedirs("data", exist_ok=True)
        np.save("data/X.npy", X)
        np.save("data/y.npy", y)
        if verbose:
            print('save data as "result/[X/y].npy"')

    if verbose:
        print("finish prepricessing!")
        print(f"data shape: {X.shape}")
    return X, y
