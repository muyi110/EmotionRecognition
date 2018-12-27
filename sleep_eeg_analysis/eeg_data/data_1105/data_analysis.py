#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def read_data(path="./"):
    delta = np.loadtxt(path+"delta.txt", skiprows=0, dtype=np.float32)
    theta = np.loadtxt(path+"theta.txt", skiprows=0, dtype=np.float32)
    lowalpha = np.loadtxt(path+"lowalpha.txt", skiprows=0, dtype=np.float32)
    highalpha = np.loadtxt(path+"highalpha.txt", skiprows=0, dtype=np.float32)
    lowbeta = np.loadtxt(path+"lowbeta.txt", skiprows=0, dtype=np.float32)
    highbeta = np.loadtxt(path+"highbeta.txt", skiprows=0, dtype=np.float32)
    lowgamma = np.loadtxt(path+"lowgamma.txt", skiprows=0, dtype=np.float32)
    midgamma = np.loadtxt(path+"midgamma.txt", skiprows=0, dtype=np.float32)
    eeg = np.loadtxt(path+"eeg.txt", skiprows=1, dtype=np.float32)[512*50:512*70] # 1min
    eeg = (eeg - min(eeg)) / (max(eeg) - min(eeg))
    return(eeg, delta, theta, lowalpha, highalpha, lowbeta, highbeta, lowgamma, midgamma)
def filter_and_fft(eeg):
    # 采样频率 512Hz
    b_delta, a_delta = signal.butter(4, [1/256, 3/256], "bandpass")           # 1-3Hz
    b_theta, a_theta = signal.butter(4, [4/256, 7/256], "bandpass")           # 4-7Hz
    b_alpha, a_alpha = signal.butter(4, [8/256, 13/256], "bandpass")    # 8-13Hz
    b_beta, a_beta = signal.butter(4, [14/256, 30/256], "bandpass")      # 14-30Hz
    b_gamma, a_gamma = signal.butter(4, [31/256, 50/256], "bandpass")   # 31-50Hz
    # 开始滤波
    delta = signal.filtfilt(b_delta, a_delta, eeg)
    theta = signal.filtfilt(b_theta, a_theta, eeg)
    alpha = signal.filtfilt(b_alpha, a_alpha, eeg)
    beta = signal.filtfilt(b_beta, a_beta, eeg)
    gamma = signal.filtfilt(b_gamma, a_gamma, eeg)
    # fft
    delta_fft = np.fft.fft(delta)
    theta_fft = np.fft.fft(theta)
    alpha_fft = np.fft.fft(alpha)
    beta_fft = np.fft.fft(beta)
    gamma_fft = np.fft.fft(gamma)
    return(delta_fft, theta_fft, alpha_fft, beta_fft, gamma_fft)

if __name__ == "__main__":
    eeg, delta, theta, lowalpha, highalpha, lowbeta, highbeta, lowgamma, midgamma = read_data()
    b, a = signal.butter(6, 25/128, "lowpass")           # 0-50Hz
    eeg = signal.filtfilt(b, a, eeg)
    # 获取 8 个频率段 fft 后的结果
    delta_fft, theta_fft, alpha_fft, beta_fft, gamma_fft = filter_and_fft(eeg)
    # 获取原始频率段 fft 结果
    eeg_fft = np.fft.fft(eeg)
    # fft 结果可视化显示
    x_eeg = np.linspace(0, 512//2, len(eeg)//2 - 1)
    x_eeg_power = np.linspace(0, 512//2, len(delta_fft)//2)
    plt.plot(x_eeg, (np.abs(eeg_fft[1:len(eeg_fft)//2])))
    plt.grid()
    plt.ylim(0, 650)
    plt.show()
    plt.subplot(231)
    plt.plot(x_eeg_power, (np.abs(delta_fft[:len(delta_fft)//2])))
    plt.grid()
    plt.title("delta(1-3Hz)")
    plt.subplot(232)
    plt.plot(x_eeg_power, (np.abs(theta_fft[:len(theta_fft)//2])))
    plt.grid()
    plt.title("theta(4-7Hz)")
    plt.subplot(233)
    plt.plot(x_eeg_power, (np.abs(alpha_fft[:len(alpha_fft)//2])))
    plt.grid()
    plt.title("alpha(8-13Hz)")
    plt.subplot(234)
    plt.plot(x_eeg_power, (np.abs(beta_fft[:len(beta_fft)//2])))
    plt.grid()
    plt.title("beta(14-30Hz)")
    plt.subplot(235)
    plt.plot(x_eeg_power, (np.abs(gamma_fft[:len(gamma_fft)//2])))
    plt.grid()
    plt.title("gamma(31-50Hz)")
    plt.show()
    # 大约 10 min 左右根据传感器采集的 8 种频率段数据分别对比，得到各个频率段能量变化情况
    delta_result=[]
    theta_result=[]
    lowalpha_result=[]
    highalpha_result=[]
    lowbeta_result=[]
    highbeta_result=[]
    lowgamma_result=[]
    midgamma_result=[]
    for index in range(len(delta)):
        sum_all = delta[index] + theta[index] + lowalpha[index] + highalpha[index] + lowbeta[index] + highbeta[index] + lowgamma[index] + midgamma[index]
        # 计算各个频率段“能量”所在比重
        delta_p = delta[index] / sum_all
        delta_result.append(delta_p)
        theta_p = theta[index] / sum_all
        theta_result.append(theta_p)
        lowalpha_p = lowalpha[index] / sum_all
        lowalpha_result.append(lowalpha_p)
        highalpha_p = highalpha[index] / sum_all
        highalpha_result.append(highalpha_p)
        lowbeta_p = lowbeta[index] / sum_all
        lowbeta_result.append(lowbeta_p)
        highbeta_p = highbeta[index] / sum_all
        highbeta_result.append(highbeta_p)
        lowgamma_p = lowgamma[index] / sum_all
        lowgamma_result.append(lowgamma_p)
        midgamma_p = midgamma[index] / sum_all
        midgamma_result.append(midgamma_p)
    assert(len(delta_result) == len(delta))
    # 开始绘制图
    x = range(len(delta_result))
    plt.plot(x, delta_result, label="delta")
    plt.plot(x, theta_result, label="theta")
    plt.plot(x, np.array(lowalpha_result) + np.array(highalpha_result), label="lowalpha")
    #plt.plot(x, highalpha_result, label="highalpha")
    plt.plot(x, np.array(lowbeta_result) + np.array(highbeta_result), label="lowbeta")
    #plt.plot(x, highbeta_result, label="highbeta")
    #plt.plot(x, lowgamma_result, label="lowgamma")
    #plt.plot(x, midgamma_result, label="midgamma")
    plt.grid()
    plt.legend()
    plt.show()
