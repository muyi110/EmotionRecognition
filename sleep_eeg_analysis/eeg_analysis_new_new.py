import numpy as np
:q
:wq
import matplotlib.pyplot as plt
import scipy.signal as signal
import PyEMD
from sklearn.decomposition import FastICA
import apen
import my_apen

# 计算自相关系数
def autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r / (variance * n)
    return np.abs(result[2])

np.random.seed(42)
# eeg = np.loadtxt("./eeg.txt", skiprows=1, dtype=np.float32)[512*1370:512*1390] # sleep
# eeg = np.loadtxt("./eeg1.txt", skiprows=1, dtype=np.float32)[512*110:512*130] # normal
eeg = np.loadtxt("./eeg_1022.txt", skiprows=1, dtype=np.float32)[512*910:512*930] # sleep
# eeg = np.loadtxt("./eeg_1022_normal.txt", skiprows=1, dtype=np.float32)[512*190:512*210] # normal
b, a = signal.butter(4, [1/512, 30/256], "bandpass")           # 0.5-30Hz
eeg = signal.filtfilt(b, a, eeg)
eeg = eeg - eeg.mean()
eeg = eeg / eeg.std()
plt.figure(5)
plt.plot(eeg)
plt.grid()
plt.title("original eeg(20S)")
plt.show()

# EMD
eemd = PyEMD.EMD()
IMFS = eemd.emd(eeg)
print("IMFS shape: ", IMFS.shape)
# 计算 IMFS 的自相关系数
temp = IMFS.copy()
auto_corr_list = []
for i in range(temp.shape[0] - 1):
    auto_corr_list.append(autocorrelation(temp[i]))
print("auto_corr: ", auto_corr_list)
# 去掉自相关系数低的通道
channels_list = []
for i in range(len(auto_corr_list)):
    if auto_corr_list[i] > 0.0:
        channels_list.append(i)
IMFS = IMFS[channels_list] 
print(IMFS.shape)
# 计算近似熵
ica_ap_list = []
ica_ap_dict = dict()
S_ = IMFS.T
for i in range((S_.T).shape[0]):
    print(i)
    ica_ap_list.append(my_apen.ApEn((S_.T)[i][:2000], 2, 0.2*(S_.T[i][:2000]).std()))
    ica_ap_dict[str(i)] = ica_ap_list[-1] 
print("ica ap: ", ica_ap_list)
ica_ap_list_temp = ica_ap_list[:]
temp = sorted(ica_ap_list_temp)
eog_channel_list = []
for key , value in ica_ap_dict.items():
    if(value < temp[-1] and value >= temp[-4]):
        eog_channel_list.append(int(key))
print("eog_channel_list: ", eog_channel_list)
# remove noise
b, a = signal.butter(4, [1/512, 30/256], "bandpass")           # 0.5-30Hz
eeg_no_eog = IMFS[eog_channel_list].sum(axis=0)
eeg_no_eog = signal.filtfilt(b, a, eeg_no_eog)

# FFT
eeg_fft = np.fft.fft(eeg)
eeg_fft_x = np.linspace(0, 256, len(eeg_fft)//2 - 1)
eeg_no_EOG_fft = np.fft.fft(eeg_no_eog)

plt.figure(2)
plt.plot(eeg_no_eog)
plt.grid()
# plt.ylim(-6,6)
plt.title("EEG remove EOG")

plt.figure(1)
plt.plot(eeg)
plt.grid()
# plt.ylim(-6,6)
plt.title("original eeg(20S)")

# plt.figure("EMD")
# for n in range(IMFS.shape[0]):
#     plt.subplot(N, 1, n+1)
#     plt.plot(IMFS[n])
#     plt.grid()
# FFT
plt.figure("FFT")
plt.plot(eeg_fft_x, np.abs(eeg_fft[1:len(eeg_fft)//2]))
plt.plot(eeg_fft_x, np.abs(eeg_no_EOG_fft[1:len(eeg_fft)//2]))
plt.grid()
plt.xlim(0, 50)
plt.title("EEG FFT Result(20S)")

# plt.figure("ICA")
# S_ = S_.T
# for i in range(S_.shape[0]):
#      plt.subplot(S_.shape[0], 1, i+1)
#      plt.plot(S_[i])
#      plt.grid()
plt.show()
