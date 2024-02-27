import numpy as np
def smooth(x, span) :
    window = span - 1
    xc = np.convolve(x, np.ones(window, dtype=int), 'valid') / window
    r = np.arange(1, window - 1, 2)
    xs = np.cumsum(x[:window - 1])[::2] / r
    xp = (np.cumsum(x[:-window:-1])[::2] / r)[::-1]
    return np.concatenate((xs, xc, xp))
