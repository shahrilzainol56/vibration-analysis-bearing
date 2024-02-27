import numpy as np
from scipy.interpolate import interp1d

# =========================================================================
# This code is programmed by System Design Optimization Lab (SDOL) at Korea
# Aerospace University (KAU)
# ============================== Input ====================================
# x: Vibration signal
# sr: Sampling frequency
# tach: Tachometer signal
# ppr: The number of pulses per 1 revolution
# Nr: The number of rotation to calculate TSA
# ============================== Output ===================================
# ta: Time synchronous average of signal x
# t: Sample time correspond to ta
# =========================================================================

def sdol_tsa(x,sr,tach,ppr,Nr=1):

    t = np.arange(0, len(x) / sr, 1 / sr)
    ppr = ppr*Nr
    if np.remainder(len(tach),ppr) == 0:
        nrev=int(len(tach)/ppr-1)
    else:
        nrev = int(np.floor(len(tach) / ppr))

    T = np.empty(shape=[nrev, 1]) #cycle start time

    for j in range(nrev):
        T[j] = tach[ppr * (j + 1)]

    T = np.insert(T, 0, tach[0])
    ind = np.where(T < t[-1]) #until maximum measurement time
    T = T[ind]

    nrev = len(T) - 1;
    N = round(min(np.diff(T)) * sr)
    resample = np.empty(shape=[nrev, N])
    for j in range(1, nrev + 1):
        tmp = (T[j] - T[j - 1]) / N; tmp = tmp * range(N)
        resample[j - 1, :] = T[j - 1] + tmp

    x = np.ravel(x, order='C')
    resx = interp1d(t, x)(resample)
    ta = np.mean(resx, axis=0)
    ta = np.reshape(ta,(len(ta),1))
    f = 1 / min(np.diff(T))
    sr_resamp = N * f
    t = range(N) / sr_resamp
    
    return ta, t

