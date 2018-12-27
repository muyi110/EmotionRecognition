import numpy as np
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
    return np.abs(result[3])

np.random.seed(42)
# eeg = np.loadtxt("./eeg_sleep.txt", skiprows=1, dtype=np.float32)[512*1700:512*1720] # sleep
# eeg = np.loadtxt("./eeg_sleep_old.txt", skiprows=1, dtype=np.float32)[512*100:512*120] # sleep
# eeg = np.loadtxt("./eeg_normal.txt", skiprows=1, dtype=np.float32)[512*75:512*95] # normal
# eeg = np.loadtxt("./eeg_learning_old.txt", skiprows=1, dtype=np.float32)[512*155:512*175] # normal
# eeg = np.loadtxt("./eeg_other_sensor_normal.txt", skiprows=1, dtype=np.float32)[512*80:512*100] # normal
eeg = np.loadtxt("./eeg_other_sensor_sleep.txt", skiprows=1, dtype=np.float32)[512*1750:512*1770] # sleep
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
# IMFS 绘图
num = IMFS.shape[0]-4
for i in range(num):
    plt.subplot(num//2+1, 2, i+1)
    fft_imf = np.fft.fft(IMFS[i])
    fft_imf_x = np.linspace(0, 256, len(fft_imf)//2 - 1)
    plt.plot(fft_imf_x, np.abs(fft_imf[1:len(fft_imf)//2]))
    plt.grid()
    plt.xlim(0,50)
plt.show()
# 计算 IMFS 的自相关系数
temp = IMFS.copy()
auto_corr_list = []
for i in range(temp.shape[0] - 1):
    auto_corr_list.append(autocorrelation(temp[i]))
print("auto_corr: ", auto_corr_list)
# 去掉自相关系数低的通道，并将其置 0
channels_list = []
for i in range(len(auto_corr_list)):
    if auto_corr_list[i] < 0.999 and auto_corr_list[i] > 0.1:
        channels_list.append(i)
# channels_list.append(IMFS.shape[0]-1)
IMFS = IMFS[channels_list] 
print(IMFS.shape)
# ICA
N = IMFS.shape[0]
ica = FastICA(n_components=N, max_iter=5000, tol=0.001)
IMFS_ICA = IMFS[:,:].T # 去掉最后一个残差分量
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
    ica_ap_list.append(my_apen.ApEn((S_.T)[i][:2000], 2, 0.2*(S_.T[i][:2000]).std()))
    ica_ap_dict[str(i)] = ica_ap_list[-1] 
print("ica ap: ", ica_ap_list)
ica_ap_list_temp = ica_ap_list[:]
temp = sorted(ica_ap_list_temp)
k_min = N - 3 # 获取 EOG 通道个数
eog_channel_list = []
for key, value in ica_ap_dict.items():
    if value == temp[-1]:
        eog_channel_list.append(int(key))
    for i in range(k_min):
        if(value == temp[i]):
            eog_channel_list.append(int(key))
print("eog_channel_list: ", eog_channel_list)
# remove noise
b, a = signal.butter(4, [1/512, 50/256], "bandpass")           # 0.5-30Hz

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
eeg_no_EOG_ = ica.inverse_transform(S_remove_EOG) * std

eeg_raw = ica.inverse_transform(S_)*std
eeg_raw = np.c_[eeg_raw,IMFS[-1,:].reshape(-1,1)]

b_delta, a_delta = signal.butter(4, [1/256, 3/256], "bandpass")           # 1-3Hz
b_theta, a_theta = signal.butter(4, [4/256, 7/256], "bandpass")           # 4-7Hz
b_alpha, a_alpha = signal.butter(4, [8/256, 13/256], "bandpass")    # 8-13Hz
b_beta, a_beta = signal.butter(4, [14/256, 30/256], "bandpass")      # 14-30Hz
delta = signal.filtfilt(b_delta, a_delta, eeg_no_EOG_.sum(axis=1))
theta = signal.filtfilt(b_theta, a_theta, eeg_no_EOG_.sum(axis=1))
alpha = signal.filtfilt(b_alpha, a_alpha, eeg_no_EOG_.sum(axis=1))
beta = signal.filtfilt(b_beta, a_beta, eeg_no_EOG_.sum(axis=1))
delta_fft = np.fft.fft(delta)
theta_fft = np.fft.fft(theta)
alpha_fft = np.fft.fft(alpha)
beta_fft = np.fft.fft(beta)

x_eeg_power = np.linspace(0, 512//2, len(delta_fft)//2)
plt.subplot(221)
plt.plot(x_eeg_power, (np.abs(delta_fft[:len(delta_fft)//2])))
plt.grid()
plt.xlim(0,30)
plt.title("delta(1-3Hz)")
plt.subplot(222)
plt.plot(x_eeg_power, (np.abs(theta_fft[:len(theta_fft)//2])))
plt.grid()
plt.xlim(0,30)
plt.title("theta(4-7Hz)")
plt.subplot(223)
plt.plot(x_eeg_power, (np.abs(alpha_fft[:len(alpha_fft)//2])))
plt.grid()
plt.xlim(0,30)
plt.title("alpha(8-13Hz)")
plt.subplot(224)
plt.plot(x_eeg_power, (np.abs(beta_fft[:len(beta_fft)//2])))
plt.grid()
plt.xlim(0,30)
plt.title("beta(14-30Hz)")
plt.show()

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
