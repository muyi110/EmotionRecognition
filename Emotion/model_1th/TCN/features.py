# -*- coding:UTF-8 -*-
######################################################################
# 此模块用于生理数据的预处理，特征提取
######################################################################
import numpy as np
import scipy.signal as signal

# 滤波器设计（采样频率 128Hz)
b_theta, a_theta = signal.butter(6, [0.0625, 0.109375], "bandpass") # 4-7Hz
b_alpha, a_alpha = signal.butter(6, [0.125, 0.203125], "bandpass")  # 8-13Hz
b_beta, a_beta = signal.butter(6, [0.21875, 0.46875], "bandpass")   # 14-30Hz
b_gamma, a_gamma = signal.butter(6, 0.484375, "highpass")           # >=31Hz
def data_filter(X, params):
    '''
    对原始输入数据进行滤波处理，提取不同频率段数据。
    -- 注意：对于 DEAP 数据集，原始预处理后的数据频率段: 4-45Hz --
    theta: 4-7Hz, alpha: 8-13Hz, beta:14-30Hz, gamma: 31-50Hz
    '''
    # # 滤波器设计（采样频率 128Hz)
    # b_theta, a_theta = signal.butter(6, [0.0625, 0.109375], "bandpass") # 4-7Hz
    # b_alpha, a_alpha = signal.butter(6, [0.125, 0.203125], "bandpass")  # 8-13Hz
    # b_beta, a_beta = signal.butter(6, [0.21875, 0.46875], "bandpass")   # 14-30Hz
    # b_gamma, a_gamma = signal.butter(6, 0.484375, "highpass")           # >=31Hz
    b_theta, a_theta, b_alpha, a_alpha, b_beta, a_beta, b_gamma, a_gamma = params
    # 滤波处理，获取各个频率段数据
    theta = signal.filtfilt(b_theta, a_theta, X).reshape(-1)
    alpha = signal.filtfilt(b_alpha, a_alpha, X).reshape(-1)
    beta = signal.filtfilt(b_beta, a_beta, X).reshape(-1)
    gamma = signal.filtfilt(b_gamma, a_gamma, X).reshape(-1)

    return (theta, alpha, beta, gamma)

def differential_entropy(X):
    '''
    计算差分熵，对于服从高斯分布的差分熵，有h(X) = 1/2 *log(2*pi*e*(std)^2)
    '''
    # 计算输入数据的方差
    var = X.var()
    # 计算差分熵
    de = 0.5*np.log(2*np.pi*np.e*var)
    return de
    
