import math

import numpy as np
from scipy.fft import fft
from scipy.signal import lfilter, butter, hilbert

# =========================================================================
# This code is programmed by System Design Optimization Lab (SDOL) at Korea
# Aerospace University (KAU)
# ============================= Input =====================================
# tsa_sig: TSA signal
# teeth: The number of teeth of gear
# sr: Sampling rate
# fr: Shaft rotational speed
# ============================= Output ====================================
# feature: Calculated feature value
# feature_name: The name of features
# =========================================================================

def gear_feat(tsa_sig, teeth, sr, fr):
    # FFT
    N = len(tsa_sig)
    X = np.abs(fft(tsa_sig, axis=0))/N*2; X = X[0:math.ceil(N/2)]
    f = np.arange(0, N)/N*sr; f = f[0:math.ceil(N/2)]
    # Find rotating speed
    ix = np.where((f>25) & (f<35))[0]; fr = f[ix][0]
    # Find GMF
    gmf = teeth*fr; cutoff = sr/N*1.2
    ix = np.where((f > gmf - cutoff) & (f < gmf + cutoff))[0]
    gmf = f[np.where(X == np.max(X[ix]))[0][0]]
    cutoff = 10
    gmf_amp = np.zeros(10)
    side_amp = np.zeros((2,6,10))
    ind_side = np.zeros((2,6,10))
    for hn in range(10):
        P = gmf*(hn+1)
        ix = np.where((f > P - cutoff) & (f < P + cutoff))[0]
        gmf_amp[hn] = np.max(X[ix])
        for sn in range(6):
            S = fr*(sn+1)
            ix_side1 = np.where((f > P - S - cutoff) & (f < P - S + cutoff))
            ix_side2 = np.where((f > P + S - cutoff) & (f < P + S + cutoff))
            side_amp[0, sn, hn] = np.max(X[ix_side1[0]])
            side_amp[1, sn, hn] = np.max(X[ix_side2[0]])

    # ==================== Calculate Residual signal ==========================
    res_sig = tsa_sig; ord = 2
    for hn in range(10):
        P = gmf*(hn+1)
        b, a = butter(ord, [(P-cutoff)/(sr/2), (P+cutoff)/(sr/2)],'bandstop')
        res_sig = lfilter(b, a, res_sig, axis=0)
    # =================== Calculate difference signal =========================
    diff_sig = res_sig
    for hn in range(10):
        P = gmf * (hn + 1)
        b, a = butter(ord, [(P - fr - cutoff) / (sr / 2), (P - fr + cutoff) / (sr / 2)], 'bandstop')
        diff_sig = lfilter(b, a, diff_sig, axis=0)
        b, a = butter(ord, [(P + fr - cutoff) / (sr / 2), (P + fr + cutoff) / (sr / 2)], 'bandstop')
        diff_sig = lfilter(b, a, diff_sig, axis=0)
    # =================== Calculate features for gear =========================
    # 1. FMO
    FM0 = (np.max(tsa_sig) - np.min(tsa_sig)) / np.sum(gmf_amp)
    # 2. SER
    SER = np.sum(np.sum(side_amp[:, :, 0])) / gmf_amp[0]
    # 3. NA4
    ress = res_sig - np.mean(res_sig)
    ress = ress.reshape(len(ress), 1)
    cur_ress = ress[:, -1]                                                        # Current signal
    N = np.shape(ress)[0]; M = np.shape(ress)[1]
    NA4 = N * np.sum(cur_ress ** 4) / ((np.sum(ress ** 2) / M) ** 2)
    # 4. FM4
    diff = diff_sig - np.mean(diff_sig)
    FM4 = len(diff) * np.sum(diff ** 4) / (np.sum(diff ** 2) ** 2)
    # 5. M6A
    M6A = (len(diff) ** 2) * np.sum(diff ** 6) / (np.sum(diff ** 2) ** 3)
    # 6. M8A
    M8A = (len(diff) ** 2) * np.sum(diff ** 8) / (np.sum(diff ** 2) ** 4)
    # 7. ER
    ER = np.sqrt(np.sum(diff_sig ** 2) / len(diff_sig)) / (np.sum(gmf_amp) + np.sum(side_amp))
    # 8. NB4
    a, b = butter(ord, [(gmf - fr) / (sr / 2), (gmf + fr) / (sr / 2)], 'bandpass')
    bp_sig = lfilter(a, b, np.ravel(tsa_sig), axis=0)
    env_bp_sig = np.abs(hilbert(bp_sig))
    s = np.array((env_bp_sig - np.mean(env_bp_sig)))
    s = s.reshape(len(s), 1)
    cur_s = s[:, -1]
    N =  np.shape(s)[0]; M = np.shape(s)[1]
    NB4 = N * np.sum(cur_s ** 4) / ((np.sum((np.sum(s ** 2, 0)), 0) / M) ** 2)

    feature = np.array([FM0, SER, NA4, FM4, M6A, M8A, ER, NB4], dtype='float64')
    feature_name = ['FM0', 'SER', 'NA4', 'FM4', 'M6A', 'M8A', 'ER', 'NB4']
    return feature, feature_name
