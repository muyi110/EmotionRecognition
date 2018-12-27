import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import PyEMD
from sklearn.decomposition import FastICA
import apen
import my_apen

np.random.seed(42)
# eeg = np.loadtxt("./eeg.txt", skiprows=1, dtype=np.float32)[512*1370:512*1390] # sleep
# eeg = np.loadtxt("./eeg1.txt", skiprows=1, dtype=np.float32)[512*110:512*130] # normal
# eeg = np.loadtxt("./eeg_1022.txt", skiprows=1, dtype=np.float32)[512*900:512*920] # sleep
eeg = np.loadtxt("./eeg_1022_normal.txt", skiprows=1, dtype=np.float32)[512*200:512*220] # sleep
b, a = signal.butter(4, [1/512, 30/256], "bandpass")           # 0.5-30Hz
eeg = signal.filtfilt(b, a, eeg)
#eeg = (eeg - eeg.min()) / (eeg.max() - eeg.min())
eeg = eeg - eeg.mean()
eeg = eeg / eeg.std()
plt.figure(5)
plt.plot(eeg)
plt.grid()
plt.title("original eeg(20S)")
plt.show()

# EMD
eemd = PyEMD.EEMD()
IMFS = eemd.eemd(eeg)
N = IMFS.shape[0] + 1
print("IMFS shape: ", IMFS.shape)
# ICA
ica = FastICA(n_components=N-2, max_iter=2000, tol=0.001)
IMFS_ICA = IMFS[:-1,:].T # 去掉最后一个残差分量
std = IMFS_ICA.std(axis=0)
IMFS_ICA = IMFS_ICA / IMFS_ICA.std(axis=0)
S_ = ica.fit_transform(IMFS_ICA)
A_ = ica.mixing_
print(A_.shape)
# 计算近似熵
ica_ap_list = []
ica_ap_dict = dict()
#ap = apen.PoolEegApEn()
for i in range((S_.T).shape[0]):
    print(i)
    #ica_ap_list.append(ap.get_pool_eeg_apen((S_.T)[i][:5000]))
    ica_ap_list.append(my_apen.ApEn((S_.T)[i][:2000], 2, 0.2*(S_.T[i][:2000]).std()))
    ica_ap_dict[str(i)] = ica_ap_list[-1] 
print("ica ap: ", ica_ap_list)
ica_ap_list_temp = ica_ap_list[:]
temp = sorted(ica_ap_list_temp)
k = []
# for i in range(2,(N-1)//2 + 1):
#     if(2*temp[i] > (temp[i-1] + temp[i+1])):
#         k.append(i)
k_min = (N-2) - 8  # 获取 EOG 通道个数
eog_channel_list = []
for key, value in ica_ap_dict.items():
    for i in range(k_min):
        if(value == temp[i]):
            eog_channel_list.append(int(key))
# for key , value in ica_ap_dict.items():
#     if(value < 0.7):
#         eog_channel_list.append(int(key))
print("eog_channel_list: ", eog_channel_list)
# remove noise
b, a = signal.butter(4, [1/512, 30/256], "bandpass")           # 0.5-30Hz

S_temp = S_.copy()
for value in eog_channel_list:
    print(S_temp[:, value].shape)
    S_temp[:, value] = np.zeros(S_temp[:, value].shape)

S_remove_EOG = S_temp
for value in range(S_remove_EOG.shape[1]):
    S_remove_EOG[:, value] = signal.filtfilt(b, a, S_remove_EOG[:, value])
    S_[:, value] = signal.filtfilt(b, a, S_[:, value])
print(S_ == S_remove_EOG)
print(S_remove_EOG.shape)
eeg_no_EOG = ica.inverse_transform(S_remove_EOG) * std
eeg_no_EOG_ = np.c_[eeg_no_EOG,IMFS[-1,:].reshape(-1,1)]

# eeg_raw = np.dot(S_, A_.T)*std
eeg_raw = ica.inverse_transform(S_)*std
eeg_raw = np.c_[eeg_raw,IMFS[-1,:].reshape(-1,1)]

# FFT
eeg_fft = np.fft.fft(eeg)
eeg_fft_x = np.linspace(0, 256, len(eeg_fft)//2 - 1)
eeg_no_EOG_fft = np.fft.fft(eeg_no_EOG_.sum(axis=1))

plt.figure(2)
plt.plot(eeg_no_EOG_.sum(axis=1))
plt.grid()
plt.ylim(-6,6)
plt.title("EEG remove EOG")

plt.figure(3)
plt.plot(eeg_raw.sum(axis=1))
plt.grid()
plt.ylim(-6,6)
plt.title("final original eeg(20S)")

plt.figure(1)
plt.plot(eeg)
plt.grid()
plt.ylim(-6,6)
plt.title("original eeg(20S)")

plt.figure("EMD")
for n in range(IMFS.shape[0]):
    plt.subplot(N, 1, n+1)
    plt.plot(IMFS[n])
    plt.grid()
# FFT
plt.figure("FFT")
plt.plot(eeg_fft_x, np.abs(eeg_fft[1:len(eeg_fft)//2]))
plt.plot(eeg_fft_x, np.abs(eeg_no_EOG_fft[1:len(eeg_fft)//2]))
plt.grid()
plt.xlim(0, 50)
plt.title("EEG FFT Result(20S)")

plt.figure("ICA")
S_ = S_.T
for i in range(S_.shape[0]):
     plt.subplot(S_.shape[0], 1, i+1)
     plt.plot(S_[i])
     plt.grid()
plt.show()
