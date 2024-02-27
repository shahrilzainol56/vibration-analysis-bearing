import pandas as pd
import os, glob
import numpy as np
import math
from scipy.fft import fft
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import io

from Functions.sdol_tsa import sdol_tsa
from Functions.TimeFeature import timefeature
from Functions.Gear_feat import gear_feat

## HS Gear
# ============================ Explanation ================================
# This code is intended to show the signal processing in the paper
# step-by-step using HS gear
# This dataset was originally downloaded at data-acoustics.com,
# which was developed by Dr. Eric Bechhoefer, but unavailable now.
# Dr. Bechhoefer gave us permission to distribute the dataset at our site,
# which is greatly appreciated.
# https://www.kau-sdol.com/kaug
# 1. Signal processing (TSA)
# 2. Feature extraction (Time features & features for gear)
# 3. Feature selection (FDR)
# Two class gear data with normal and fault
# =========================================================================
plt.rc('font', size=13)
os.chdir('Data_repository/HS_Gear')                             # Input data folder directory
fr = 30                                                         # Rotating speed
Nr = 1                                                          # Number of rotations for TSA
# ===================== 1. Load normal data & TSA =========================
# Normal
os.chdir('data2')
file_list = os.listdir()
N_normal = len(file_list)
ix_example = 1
normal = []
for ix_file in range(N_normal):
    sr = int(io.loadmat(file_list[ix_file])['sr'])
    teeth = int(io.loadmat(file_list[ix_file])['teeth'])
    ppr = int(io.loadmat(file_list[ix_file])['ppr'])
    gs = io.loadmat(file_list[ix_file])['gs']
    tach = io.loadmat(file_list[ix_file])['tach']
    tix = tach[ppr-1]
    cyc = math.floor(sr*tix)
    ta,t = sdol_tsa(gs, sr, tach, ppr)                          # TSA
    normal.append(ta)
    # Example to ililustrate effects of TSA
    if ix_file == ix_example:
        x = gs                                                  # Raw data of example
        t2 = t                                                  # Time vector for TSA signal of example

os.chdir("..")

# Fault
os.chdir('data1')
file_list = os.listdir()
N_fault = len(file_list)
fault = []
for i in file_list:
    gs = io.loadmat(i)['gs']
    tach = io.loadmat(i)['tach']
    tix = tach[ppr-1]
    cyc = math.floor(sr*tix)
    ta,t = sdol_tsa(gs,sr,tach,ppr)                             # TSA
    fault.append(ta)

# Time domain signal <Figure 3(a)>
t = np.arange(0, t2[-1], 1 / sr); N = len(t)
x1 = x[0:N]; x2 = normal[ix_example-1]
plt.figure(1)
plt.subplot(211); plt.plot(t, x1); plt.xlim(0,); plt.title('Raw data')
plt.subplot(212); plt.plot(t2, x2); plt.xlim(0,); plt.title('TSA data'); plt.xlabel('Time (s)')
plt.tight_layout()

# Frequency domain signal <Figure 3(b)>
f = np.arange(0, N)/N*sr; f = f[0:math.ceil(N/2)]
X1 = np.abs(fft(x1, axis=0))/N*2; X1 = X1[0:math.ceil(N/2)]
X2 = np.abs(fft(x2, axis=0))/N*2; X2 = X2[0:math.ceil(N/2)]
plt.figure(2)
plt.subplot(211); plt.stem(f, X1, basefmt=" "); plt.title('Raw data'); plt.xlim(400, 1400); plt.ylim(0, 0.35)
plt.subplot(212); plt.stem(f, X2, basefmt=" "); plt.title('TSA data'); plt.xlim(400, 1400); plt.ylim(0, 0.2)
plt.xlabel('Frequency(Hz)')
plt.tight_layout()
plt.show()

##
# ======================== 2. Feature extraction ==========================
plt.rc('font', size=16)
# Normal
features_normal = np.zeros((6,19))
for ix in range(N_normal):
    tmp1, fn_time = timefeature(normal[ix])
    tmp2, fn_gear = gear_feat(normal[ix], teeth, sr, fr)
    features_normal[ix, :] = np.hstack([tmp1, tmp2])

# Fault
features_fault = np.zeros((11,19))
for ix in range(N_fault):
    tmp1, fn_time = timefeature(fault[ix])
    tmp2, fn_gear = gear_feat(fault[ix], teeth, sr, fr)
    features_fault[ix, :] = np.hstack([tmp1, tmp2])

# Normalizing
features = np.concatenate([features_normal, features_fault], axis=0)
m = np.mean(features, axis=0); s = np.std(features, ddof=1, axis=0)
features = (features-m)/s
ix_normal = np.arange(0,N_normal); ix_fault = np.arange(N_normal, N_normal+N_fault)

# Time domain features of HS gear <Figure 8>
plt.figure(1, figsize=(12, 6))
for j in range(len(fn_time)):
    plt.scatter(j * np.ones(N_normal), features[ix_normal, j], c='b', marker='o', s=100)
    plt.scatter(j * np.ones(N_fault), features[ix_fault, j], c='r', marker='x', s=100)
plt.xticks(range(len(fn_time)), fn_time); plt.xlim(-1, len(fn_time))
plt.title('Time domain features of HS gear');
plt.ylabel('Normalized value'); plt.grid()

# Specific features of HS gear <Figure 9>
plt.figure(2, figsize=(12, 6))
for j in range(len(fn_time), len(fn_time)+len(fn_gear)):
    ix = j-len(fn_time)
    plt.scatter(ix * np.ones(N_normal), features[ix_normal, j], c='b', marker='o', s=100)
    plt.scatter(ix * np.ones(N_fault), features[ix_fault, j], c='r', marker='x', s=100)
plt.xticks(range(len(fn_gear)), fn_gear); plt.xlim(-1, len(fn_gear))
plt.title('Features for gear of HS gear')
plt.ylabel('Normalized value'); plt.grid()
plt.show()

# ======================== 3. Features selection ==========================
plt.rc('font', size=13)
feature_name = fn_time + fn_gear
features_normal = features[ix_normal,:]; features_fault = features[ix_fault,:]
m_normal = np.mean(features_normal, axis=0); m_fault = np.mean(features_fault, axis=0)
s_normal = np.std(features_normal, ddof=1, axis=0); s_fault = np.std(features_fault, ddof=1, axis=0)
fdr = (m_normal-m_fault)**2/(s_normal**2 + s_fault**2)                              # FDR

# Sorting
ix = np.argsort(fdr)[::-1]
fdr = fdr[ix]

# Best3 <Table 11>
Top3 = [1,2,3]
Feature = feature_name[ix[0]], feature_name[ix[1]], feature_name[ix[2]]
FDR_value = [fdr[0], fdr[1], fdr[2]]
T1 = pd.DataFrame([Top3, Feature, FDR_value])

# Worst3 <Table 11>
Bottom3 = [1,2,3]
Feature = feature_name[ix[-1]], feature_name[ix[-2]], feature_name[ix[-3]]
FDR_value = [fdr[-1], fdr[-2], fdr[-3]]
T2 = pd.DataFrame([Bottom3, Feature, FDR_value])

# High FDR value <Figure 13(a)>
x1 = np.arange(m_normal[ix[0]] - 3 * s_normal[ix[0]], m_normal[ix[0]] + 3 * s_normal[ix[0]],0.01)
pdf1=norm.pdf(x1,m_normal[ix[0]], s_normal[ix[0]])
x2 = np.arange(m_fault[ix[0]] - 3 * s_fault[ix[0]], m_fault[ix[0]] + 3 * s_fault[ix[0]],0.01)
pdf2=norm.pdf(x2,m_fault[ix[0]],s_fault[ix[0]])
plt.figure(1)
plt.plot(x1, pdf1, c='b'); plt.plot(x2, pdf2, c='r',ls= '--')
plt.title(f'{feature_name[ix[0]]} (FDR = {fdr[0]:.4f})')
plt.legend(['Normal','Fault']); plt.ylim(0,)

# Low FDR value <Figure 13(b)>
x1 = np.arange(m_normal[ix[-1]] - 3 * s_normal[ix[-1]], m_normal[ix[-1]] + 3 * s_normal[ix[-1]],0.01)
pdf1=norm.pdf(x1,m_normal[ix[-1]], s_normal[ix[-1]])
x2 = np.arange(m_fault[ix[-1]] - 3 * s_fault[ix[-1]], m_fault[ix[-1]] + 3 * s_fault[ix[-1]],0.01)
pdf2=norm.pdf(x2,m_fault[ix[-1]], s_fault[ix[-1]])
plt.figure(2)
plt.plot(x1, pdf1, c='b'); plt.plot(x2, pdf2, c='r',ls= '--')
plt.title(f'{feature_name[ix[-1]]} (FDR = {fdr[-1]:.8f})')
plt.legend(['Normal','Fault']); plt.ylim(0,)
plt.show()

