# -*- coding:UTF-8 -*-
#########################################################
# 此模块用于数据的预处理，特征提取. -----------2018-11-26
#########################################################
import numpy as np
import scipy.signal as signal

def filter_with_stft(x, fs=128, window='hann', nperseg=128, noverlap=None, nfft=256):
    '''
      利用短时傅里叶变换进行滤波，得到 4 个子频带
      theta(4-7Hz), alpha(8-13Hz), beta(14-30Hz), gamma(31-50Hz)
    '''
    # 短时傅里叶变化，窗函数长度 1s
    f, t, Zxx = signal.stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    # 下面提取 theta 频率段的频域信息
    theta_index_1 = f >= 3.5
    theta_index_2 = f <= 7.5
    theta_index = theta_index_1 == theta_index_2
    theta_index = theta_index.reshape(-1, 1)
    theta_index = np.c_[tuple([theta_index]*t.shape[0])]
    theta_freq = np.where(theta_index, Zxx, 0)
    # 逆变换，得到 theta 频率段时域信号
    _, rec_theta = signal.istft(theta_freq, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    # 下面提取 alpha 频率段的频域信息
    alpha_index_1 = f >= 7.5
    alpha_index_2 = f <= 13.5
    alpha_index = alpha_index_1 == alpha_index_2
    alpha_index = alpha_index.reshape(-1, 1)
    alpha_index = np.c_[tuple([alpha_index]*t.shape[0])]
    alpha_freq = np.where(alpha_index, Zxx, 0)
    # 逆变换，得到 alpha 频率段时域信号
    _, rec_alpha = signal.istft(alpha_freq, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    # 下面提取 beta 频率段的频域信息
    beta_index_1 = f >= 13.5
    beta_index_2 = f <= 30.5
    beta_index = beta_index_1 == beta_index_2
    beta_index = beta_index.reshape(-1, 1)
    beta_index = np.c_[tuple([beta_index]*t.shape[0])]
    beta_freq = np.where(beta_index, Zxx, 0)
    # 逆变换，得到 beta 频率段时域信号
    _, rec_beta = signal.istft(beta_freq, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    # 下面提取 gamma 频率段的频域信息
    gamma_index_1 = f >= 30.5
    gamma_index_2 = f <= 50.0
    gamma_index = gamma_index_1 == gamma_index_2
    gamma_index = gamma_index.reshape(-1, 1)
    gamma_index = np.c_[tuple([gamma_index]*t.shape[0])]
    gamma_freq = np.where(gamma_index, Zxx, 0)
    # 逆变换，得到 theta 频率段时域信号
    _, rec_gamma = signal.istft(gamma_freq, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    assert (len(rec_theta) == len(x))
    assert (len(rec_alpha) == len(x))
    assert (len(rec_beta) == len(x))
    assert (len(rec_gamma) == len(x))
    
    return (rec_theta, rec_alpha, rec_beta, rec_gamma)

def differential_entropy(data_1s, low_feq, high_feq):
    '''
      从子频率带计算对应的差分熵
    '''
    data = data_1s * signal.get_window('hann', 128)
    fft = np.fft.rfft(data, 256)
    freq = np.linspace(0, 64, 129)
    index_1 = freq >= (low_feq - 0.5)
    index_2 = freq <= (high_feq + 0.5)
    index = index_1 == index_2
    N = sum(index)
    psd = np.sum(np.abs(fft[index])**2) / N
    de = np.log2(psd)
    return de
