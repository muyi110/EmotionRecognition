# -*- coding=UTF-8 -*-
################################################################################
# 对数据或特征的平滑处理
# 滑动平均法
################################################################################
import numpy as np
import matplotlib.pyplot as plt

def moving_average(X, seq_length, windows=6):
    '''
      对数据做滑动平均处理
      windows: 滑动窗口的大小(s)
      X: shape=(samples, features, seq_length)
    '''
    X = np.array(X)
    numbers_one_trial = 60 - seq_length + 1 # 一个实验划分的样本数
    number_samples = X.shape[0]
    assert (number_samples % numbers_one_trial == 0)
    numbers_trial = number_samples // numbers_one_trial # 获取共有多少个实验(一个实验是 60s )
    trial_list = []
    result_list = []
    for trial in range(numbers_trial):
        trial_list.append(X[trial*numbers_one_trial:(trial+1)*numbers_one_trial,:,:])
    # 开始对每一个实验进行滑动平均处理
    for trial in trial_list:
        # 每一个 trial 的 shape=(numbers_one_trial, features, seq_length)
        # 开始将 trial 拼接为 60s 长的 shape=(features, 60)
        temp = []
        for i in range(1, numbers_one_trial):
            temp.append(trial[i, :, -1])
        one_trial_feqtures = np.c_[tuple(temp)]
        one_trial_feqtures = np.c_[trial[0, :, :], one_trial_feqtures]
        assert (one_trial_feqtures.shape == (128, 60))
        # 根据窗口大小划分的样本总个数
        numbers = (60 - windows) + 1
        temp_X = one_trial_feqtures.copy()
        for number in range(numbers):
            one_trial_feqtures[:,number+windows-1] = temp_X[:, number:(number+windows)].mean(axis=1)
        # 开始处理最开始一个窗口中的数据
        for i in range(windows-1):
            one_trial_feqtures[:, i] = temp_X[:, 0:i+1].mean(axis=1)
        # # 将特征进行 0-1 处理（沿着时间轴）
        # _max = one_trial_feqtures.max(axis=1).reshape(-1,1)
        # _min = one_trial_feqtures.min(axis=1).reshape(-1,1)
        # one_trial_feqtures = (one_trial_feqtures-_min) / _max
        # 将滤波后的 60s 长按照原来时间窗口，还原
        for i in range(numbers_one_trial):
            result_list.append(one_trial_feqtures[:, i:(i+seq_length)])

    assert (np.array(result_list).shape == X.shape)
    return result_list
