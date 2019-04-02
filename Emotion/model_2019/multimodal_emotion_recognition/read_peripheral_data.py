# -*- coding:UTF-8 -*-
import os
import numpy as np

def _get_samples_data(people_list, path, windows=9, overlapping=8):
    samples_dirs = os.listdir(path) # 目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    EOG_datas_list = [] # 获取最终的样本
    EOG_labels_list = [] # 对应的样本
    EMG_datas_list = [] # 获取最终的样本
    EMG_labels_list = [] # 对应的样本
    GSR_datas_list = [] # 获取最终的样本
    GSR_labels_list = [] # 对应的样本
    RSP_datas_list = [] # 获取最终的样本
    RSP_labels_list = [] # 对应的样本
    BLV_datas_list = [] # 获取最终的样本
    BLV_labels_list = [] # 对应的样本
    TMR_datas_list = [] # 获取最终的样本
    TMR_labels_list = [] # 对应的样本
    for people in people_list:
        datas_0 = np.load(file_path[people]+"/datas_0.npy")
        labels_0 = np.load(file_path[people]+"/labels_0.npy") 
        datas_1 = np.load(file_path[people]+"/datas_1.npy")
        labels_1 = np.load(file_path[people]+"/labels_1.npy") 
        datas_2 = np.load(file_path[people]+"/datas_2.npy")
        labels_2 = np.load(file_path[people]+"/labels_2.npy") 
        datas_3 = np.load(file_path[people]+"/datas_3.npy")
        labels_3 = np.load(file_path[people]+"/labels_3.npy") 
        datas_4 = np.load(file_path[people]+"/datas_4.npy")
        labels_4 = np.load(file_path[people]+"/labels_4.npy") 
        datas_5 = np.load(file_path[people]+"/datas_5.npy")
        labels_5 = np.load(file_path[people]+"/labels_5.npy") 
        datas_6 = np.load(file_path[people]+"/datas_6.npy")
        labels_6 = np.load(file_path[people]+"/labels_6.npy") 
        datas_7 = np.load(file_path[people]+"/datas_7.npy")
        labels_7 = np.load(file_path[people]+"/labels_7.npy") 
        # 各个外周生理信号传感器数据组合
        EOG_datas = np.concatenate((datas_0, datas_1), axis=1)
        assert(EOG_datas.shape == (40, 20, 60))
        EOG_labels = labels_0
        EMG_datas = np.concatenate((datas_2, datas_3), axis=1)
        assert(EMG_datas.shape == (40, 20, 60))
        EMG_labels = labels_2
        GSR_datas = datas_4
        GSR_labels = labels_4
        RSP_datas = datas_5
        RSP_labels = labels_5
        BLV_datas = datas_6
        BLV_labels = labels_6
        TMR_datas = datas_7
        TMR_labels = labels_7
        assert(TMR_labels.shape == (40, ))
        for trial_num in range(40):
            EOG_data = EOG_datas[trial_num]
            EOG_label = EOG_labels[trial_num]
            assert (EOG_data.shape == (20, 60))
            # 这里可以添加滤波处理
            EOG_numbers = 60 - 20 + 1 # 20s的滑动窗口
            EOG_temp_data = EOG_data.copy()
            for number in range(EOG_numbers):
                EOG_data[:, number+20-1] = EOG_temp_data[:, number:(number+20)].mean(axis=1)
            for i in range(20-1):
                EOG_data[:, i] = EOG_temp_data[:, 0:i+1].mean(axis=1)
            # 特征进行0-1处理
            _max = EOG_data.max(axis=0).reshape(1, -1)
            _min = EOG_data.min(axis=0).reshape(1, -1)
            EOG_data = (EOG_data - _min) / (_max - _min)            
            assert (EOG_data.shape == (20, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = EOG_data[:, iterator*step:iterator*step+windows]
                EOG_datas_list.append(data_slice)
                EOG_labels_list.append(EOG_label)
        for trial_num in range(40):
            EMG_data = EMG_datas[trial_num]
            EMG_label = EMG_labels[trial_num]
            assert (EMG_data.shape == (20, 60))
            # 这里可以添加滤波处理
            EMG_numbers = 60 - 20 + 1 # 20s的滑动窗口
            EMG_temp_data = EMG_data.copy()
            for number in range(EMG_numbers):
                EMG_data[:, number+20-1] = EMG_temp_data[:, number:(number+20)].mean(axis=1)
            for i in range(20-1):
                EMG_data[:, i] = EMG_temp_data[:, 0:i+1].mean(axis=1)
            # 特征进行0-1处理
            _max = EMG_data.max(axis=0).reshape(1, -1)
            _min = EMG_data.min(axis=0).reshape(1, -1)
            EMG_data = (EMG_data - _min) / (_max - _min)            
            assert (EMG_data.shape == (20, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = EMG_data[:, iterator*step:iterator*step+windows]
                EMG_datas_list.append(data_slice)
                EMG_labels_list.append(EMG_label)
        for trial_num in range(40):
            GSR_data = GSR_datas[trial_num]
            GSR_label = GSR_labels[trial_num]
            assert (GSR_data.shape == (10, 60))
            # 这里可以添加滤波处理
            GSR_numbers = 60 - 20 + 1 # 20s的滑动窗口
            GSR_temp_data = GSR_data.copy()
            for number in range(GSR_numbers):
                GSR_data[:, number+20-1] = GSR_temp_data[:, number:(number+20)].mean(axis=1)
            for i in range(20-1):
                GSR_data[:, i] = GSR_temp_data[:, 0:i+1].mean(axis=1)
            # 特征进行0-1处理
            _max = GSR_data.max(axis=0).reshape(1, -1)
            _min = GSR_data.min(axis=0).reshape(1, -1)
            GSR_data = (GSR_data - _min) / (_max - _min)            
            assert (GSR_data.shape == (10, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = GSR_data[:, iterator*step:iterator*step+windows]
                GSR_datas_list.append(data_slice)
                GSR_labels_list.append(GSR_label)
        for trial_num in range(40):
            RSP_data = RSP_datas[trial_num]
            RSP_label = RSP_labels[trial_num]
            assert (RSP_data.shape == (10, 60))
            # 这里可以添加滤波处理
            RSP_numbers = 60 - 20 + 1 # 20s的滑动窗口
            RSP_temp_data =RSP_data.copy()
            for number in range(RSP_numbers):
                RSP_data[:, number+20-1] = RSP_temp_data[:, number:(number+20)].mean(axis=1)
            for i in range(20-1):
                RSP_data[:, i] = RSP_temp_data[:, 0:i+1].mean(axis=1)
            # 特征进行0-1处理
            _max = RSP_data.max(axis=0).reshape(1, -1)
            _min = RSP_data.min(axis=0).reshape(1, -1)
            RSP_data = (RSP_data - _min) / (_max - _min)            
            assert (RSP_data.shape == (10, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = RSP_data[:, iterator*step:iterator*step+windows]
                RSP_datas_list.append(data_slice)
                RSP_labels_list.append(RSP_label)
        for trial_num in range(40):
            BLV_data = BLV_datas[trial_num]
            BLV_label = BLV_labels[trial_num]
            assert (BLV_data.shape == (10, 60))
            # 这里可以添加滤波处理
            BLV_numbers = 60 - 20 + 1 # 20s的滑动窗口
            BLV_temp_data = BLV_data.copy()
            for number in range(BLV_numbers):
                BLV_data[:, number+20-1] = BLV_temp_data[:, number:(number+20)].mean(axis=1)
            for i in range(20-1):
                BLV_data[:, i] = BLV_temp_data[:, 0:i+1].mean(axis=1)
            # 特征进行0-1处理
            _max = BLV_data.max(axis=0).reshape(1, -1)
            _min = BLV_data.min(axis=0).reshape(1, -1)
            BLV_data = (BLV_data - _min) / (_max - _min)            
            assert (BLV_data.shape == (10, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = BLV_data[:, iterator*step:iterator*step+windows]
                BLV_datas_list.append(data_slice)
                BLV_labels_list.append(BLV_label)
        for trial_num in range(40):
            TMR_data = TMR_datas[trial_num]
            TMR_label = TMR_labels[trial_num]
            assert (TMR_data.shape == (10, 60))
            # 这里可以添加滤波处理
            TMR_numbers = 60 - 20 + 1 # 20s的滑动窗口
            TMR_temp_data = TMR_data.copy()
            for number in range(TMR_numbers):
                TMR_data[:, number+20-1] = TMR_temp_data[:, number:(number+20)].mean(axis=1)
            for i in range(20-1):
                TMR_data[:, i] = TMR_temp_data[:, 0:i+1].mean(axis=1)
            # 特征进行0-1处理
            _max = TMR_data.max(axis=0).reshape(1, -1)
            _min = TMR_data.min(axis=0).reshape(1, -1)
            TMR_data = (TMR_data - _min) / (_max - _min)            
            assert (TMR_data.shape == (10, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = TMR_data[:, iterator*step:iterator*step+windows]
                TMR_datas_list.append(data_slice)
                TMR_labels_list.append(TMR_label)
    
        result = (EOG_datas_list, EOG_labels_list, EMG_datas_list, EMG_labels_list, GSR_datas_list, GSR_labels_list, 
                  RSP_datas_list, RSP_labels_list, BLV_datas_list, BLV_labels_list, TMR_datas_list, TMR_labels_list)
        return result

def read_data(people_list, windows=9, overlapping=8, classify_object_name=0):
    # datas 和 labels 都是 list. datas 中的每一项都是 shape=(80, windows) 的 np.array
    if classify_object_name == 0:
        path = "../common/peripheral_features/valence_peripheral/"
    elif classify_object_name == 1:
        path = "../common/peripheral_features/arousal_peripheral/"
    return _get_samples_data(people_list, path, windows, overlapping)
