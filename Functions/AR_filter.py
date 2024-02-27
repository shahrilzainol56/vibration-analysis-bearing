from scipy.stats import kurtosis
from scipy.signal import lfilter
from nitime.algorithms.autoregressive import AR_est_YW
import numpy as np

# ============================== Input ==========================================
# x: Vibration signal
# maxK: The maximum filter order to calculate iterations
# ix: Plot the figure when the value is entered
# ============================== Output =========================================
# ardata: Residual signal
# p: Selected filter order
# ===============================================================================

def ar_filter(x, maxk, ix = 0):
    pp = np.arange(1,maxk,10); N = len(pp)
    k = np.zeros(N)
    for ix in range(0, N):
        a1 = AR_est_YW(x, pp[ix], rxx=None)                 # AR filter parameter
        xp = lfilter(np.append(0, a1[0]), [1], x, axis=0)
        xn = x - xp
        k[ix] = kurtosis(xn, fisher=False)

    ind = np.where(k == np.max(k))[0][0]
    p = pp[ind]
    a1 = AR_est_YW(x, p)[0]
    xp = lfilter(np.append(0, a1), [1], x, axis=0)
    ardata = x - xp                                         # Residual signal
    return ardata, p
