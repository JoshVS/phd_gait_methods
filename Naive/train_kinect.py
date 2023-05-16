from dataset import NaiveKinectDataset
from naive_classifier import create_classifier
import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


from scipy.signal import savgol_filter, find_peaks
import numpy as np
from scipy.optimize import curve_fit

def test_fun(x, dist, amp, omega, phi):
    return dist + amp * np.cos(omega * x + phi)

def polyfunc(x, args):
    ret = np.zeros(x.shape)
    for i, a in enumerate(args[::-1]):
        ret += a * x ** i
    return ret

ds = NaiveKinectDataset(max_samples=10)
ds.show_video(0)
d, l, r = ds.ankle_distances(2, return_positions=True)



x = np.arange(len(d))

# remove = 400

# f = np.fft.fft(d, len(d) + remove)
# d_trans = np.fft.ifft(f[remove:])
# print(len(d_trans), len(d))

# window = 4
# d_trans = []
# for i in range(len(d)):
#     lower = 0 if i < window else i - window
#     upper = -1 if i + window >= len(d) else i + window
#     r = upper - lower
#     d_trans.append(sum(d[lower:upper]) / r)

# params, _ = curve_fit(test_fun, x, d_trans)

max_val = 50

plt.figure()
# plt.plot(d[:max_val])
# plt.plot(d_trans)
# plt.plot(test_fun(x, *params))
plt.plot(l[:max_val])
plt.plot(r[:max_val])
plt.show()