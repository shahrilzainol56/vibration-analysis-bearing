import numpy as np
import pandas as pd

# =========================================================================
# This code is programmed by Seokgoo Kim in 2018
# ============================= Input =====================================
# data(NxM) : N samples of M features
# label(Nx1) : Labels of N samples
# ============================= Output ====================================
# J3 : Calculated J3 value
# =========================================================================
def ScattMat(data, label):
    label = np.array(label); class_ = pd.unique(label)
    M = len(class_); N = len(label)
    if np.shape(data)[0] != len(label):
        data = data.T
    sw = 0; sb = 0
    mu = np.mat(np.mean(data, axis=0))                                                                                  # Global mean vector
    for i in range(0, M):
        temp = data[label == class_[i]]
        s = 1 / (len(temp) - 1) * (temp - np.mat(np.mean(temp, axis=0))).T*(temp - np.mat(np.mean(temp, axis=0)))       # Cov. matrix for class i
        sw = sw + len(temp) / N * s                                                                                     # Within-class scatter matrix
        sb = sb + len(temp) / N * (np.mat(np.mean(temp, axis=0)) - mu).T * (np.mat(np.mean(temp, axis=0)) - mu)         # Between class scatter matrix

    sm = sw + sb                                                                                                        # Mixture scatter matrix
    J3 = np.trace(np.linalg.inv(sw) * sm)                                                                               # J3 value
    return J3
