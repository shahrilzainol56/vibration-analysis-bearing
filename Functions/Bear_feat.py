import math

import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft

# =========================================================================
# This code is programmed by System Design Optimization Lab (SDOL) at Korea
# Aerospace University (KAU)
# ============================== Input ====================================
# x : Input data
# fs : Sampling frequency
# bff : Bearing fault frequency 1x4 matrix [bpfo,bpfi,ftf,bsf]
# cutoff : Bandwidth to find amplitude of fault frequency
# ============================== Output ===================================
# feature : Calculated feature value
# feature_name: The name of features
# =========================================================================
def Bear_feat(x, fs, bff, cutoff):
    # FFT
    x= np.abs(hilbert(x)); x = x -np.mean(x); N = len(x)
    X = np.abs(fft(x))/N*2; X = X[0:math.ceil(N/2)]
    f = np.arange(0,N)/N*fs; f = f[0:math.ceil(N/2)]
    # Find amplitude at bearing fault frequency
    bpfo_ind = np.nonzero((bff[0] - cutoff < f) & (f < bff[0] + cutoff)); bpfo_amp = max(X[bpfo_ind])
    bpfi_ind = np.nonzero((bff[1] - cutoff < f) & (f < bff[1] + cutoff)); bpfi_amp = max(X[bpfi_ind])
    ftf_ind = np.nonzero((bff[2] - cutoff < f )& (f < bff[2] + cutoff)); ftf_amp = max(X[ftf_ind])
    bsf_ind = np.nonzero((bff[3] - cutoff < f) & (f < bff[3] + cutoff)); bsf_amp = max(X[bsf_ind])
    feature = np.array([bpfo_amp, bpfi_amp,ftf_amp,bsf_amp],dtype='float64')
    feature_name = ['BPFO', 'BPFI', 'FTF','BSF']
    return feature, feature_name
