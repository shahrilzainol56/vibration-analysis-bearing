## IMS_Bearing
import os

import numpy as np
from scipy import io
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from Functions.AR_filter import ar_filter
from Functions.Bear_feat import Bear_feat
from Functions.TimeFeature import timefeature
from Functions.Skbp import skbp

# ========================= Explanation ===================================
# This code describes the feature selection for cabstone_prognosis using IMS bearing
# dataset
# Used data is IMS bearing dataset available at the link below
# https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
# Authors converted the data into mat file
# 1. Feature extraction (Time features, Features for bearing)
# 2. Feature selection for cabstone_prognosis
# =========================================================================

## IMS Bearing
plt.rc('font', size=13)
os.chdir('Data_repository/IMS_bearing')
file_list = os.listdir()
N_file = len(file_list)
# 1. Feature extraction
features = np.zeros((N_file, 15))
fs = 20480; cutoff = 3; fr = 2000/60
bff = np.array([7.0921, 8.9079, 0.4433, 4.1975])
for ix_file in range(N_file):
    x = io.loadmat(file_list[ix_file])['x']
    tmp1, fn_time = timefeature(x)                                      # Extract time domain features
    xr = ar_filter(np.ravel(x), 700)[0]
    xb = skbp(xr, fs, 4)
    tmp2, fn_bearing = Bear_feat(xb, fs, bff*fr, cutoff)                # Extract features for bearing
    features[ix_file, :] = np.concatenate([tmp1, tmp2])
feature_name = fn_time + fn_bearing

# 2. Feature selection for cabstone_prognosis
cycle = np.arange(0,N_file); w = np.mat([0.33, 0.33, 0.33])

# Normalizing
m = np.mean(features, axis=0); s = np.std(features, ddof=1, axis=0)
features = (features - m) / s

# Smoothing
from Functions.Smooth import smooth
features_smth = np.zeros_like(features)
for ix in range(np.shape(features)[1]):
    features_smth[:,ix] = smooth(features[:,ix], 50)

# Monotonicity
mon = np.abs(np.sum(np.diff(features_smth, axis=0)>0, axis=0)-np.sum(np.diff(features_smth, axis=0)<0, axis=0))/(N_file-1)

# Trendability
tre = np.abs(spearmanr(cycle, features_smth))[0,1:,0]

# Robustness
rob = np.zeros(15)
for ix in range(np.shape(features)[1]):
    rob[ix] = np.mean(np.exp(-np.abs(features[:,ix]-features_smth[:,ix]/features[:,ix])))
cri = np.ravel(np.c_[mon, tre, rob] * w.T)
ix = np.argsort(cri)[::-1]
cri_sort = cri[ix]

# Feature selection criteria <Figure 15(a)>
plt.figure(1, figsize=(12, 6)); plt.ylim(0,0.5)
plt.bar(range(0, 15), cri); plt.title('Feature selection critera')
plt.xticks(range(0, 15), feature_name)

# High criteria <Figure 15(b)>
plt.figure(2); plt.plot(cycle, features[:,ix[0]])
plt.xlabel('Cycle'); plt.ylabel('Normalized value'); plt.xlim(0,1000)
plt.title(f'{feature_name[ix[0]]} (Criteria = {cri_sort[0]:.5f})')

# Low criteria <Figure 15(c)>
plt.figure(3); plt.plot(cycle, features[:,ix[-1]])
plt.xlabel('Cycle'); plt.ylabel('Normalized value'); plt.xlim(0,1000)
plt.title(f'{feature_name[ix[-1]]} (Criteria = {cri_sort[-1]:.5f})')
plt.show()
