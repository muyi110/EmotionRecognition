# -*- coding:UTF-8 -*-
################################################
# 读取原始的外周生理信号数据作为输入
################################################
import os
import sys
sys.path.append("../../common/")
from read_peripheral_physiological_signal_data import get_data
import numpy as np

def read_data(people_list, windows=9, overlapping=8, classify_object_name=0):
    assert(len(people_list) == 1)
    trial_list = list(range(40))
    datas, labels = get_data(people_list, trial_list, classify_object_name=classify_object_name)
    datas = np.array(datas)
    labels = np.array(labels)
    assert(datas.shape == (len(people_list)*40, 8, 7680))
    assert(labels.shape == (len(people_list)*40, ))
    EOG_datas_list = []
    EOG_labels_list = []
    EMG_datas_list = []
    EMG_labels_list = []
    GSR_datas_list = []
    GSR_labels_list = []
    RSP_datas_list = []
    RSP_labels_list = []
    BLV_datas_list = []
    BLV_labels_list = []
    TMR_datas_list = []
    TMR_labels_list = []
    # 将各个通道数据单独提取出来
    datas_0 = datas[:, 0, :].reshape(40, 60, 128)
    datas_0 = datas_0.transpose((0, 2, 1))
    datas_1 = datas[:, 1, :].reshape(40, 60, 128)
    datas_1 = datas_1.transpose((0, 2, 1))
    datas_2 = datas[:, 2, :].reshape(40, 60, 128)
    datas_2 = datas_2.transpose((0, 2, 1))
    datas_3 = datas[:, 3, :].reshape(40, 60, 128)
    datas_3 = datas_3.transpose((0, 2, 1))
    datas_4 = datas[:, 4, :].reshape(40, 60, 128)
    datas_4 = datas_4.transpose((0, 2, 1))
    datas_5 = datas[:, 5, :].reshape(40, 60, 128)
    datas_5 = datas_5.transpose((0, 2, 1))
    datas_6 = datas[:, 6, :].reshape(40, 60, 128)
    datas_6 = datas_6.transpose((0, 2, 1))
    datas_7 = datas[:, 7, :].reshape(40, 60, 128)
    datas_7 = datas_7.transpose((0, 2, 1))
    # 外周生理信号传感器数据组合
    EOG_datas = np.concatenate((datas_0, datas_1), axis=1)
    assert(EOG_datas.shape == (40, 256, 60))
    EMG_datas = np.concatenate((datas_2, datas_3), axis=1)
    assert(EMG_datas.shape == (40, 256, 60))
    GSR_datas = datas_4
    assert(GSR_datas.shape == (40, 128, 60))
    RSP_datas = datas_5
    assert(RSP_datas.shape == (40, 128, 60))
    BLV_datas = datas_6
    assert(BLV_datas.shape == (40, 128, 60))
    TMR_datas = datas_7
    assert(TMR_datas.shape == (40, 128, 60))
    for trial_num in range(40):
        EOG_data = EOG_datas[trial_num]
        EOG_label = labels[trial_num]
        assert(EOG_data.shape == (256, 60))
        # 0-1化处理
        _max = EOG_data.max(axis=0).reshape(1, -1)
        _min = EOG_data.min(axis=0).reshape(1, -1)
        EOG_data = (EOG_data - _min) / (_max - _min)
        assert(EOG_data.shape == (256, 60))
        step = windows - overlapping
        iterator_num = int((60 - windows) / step + 1)
        for iterator in range(iterator_num):
            data_slice = EOG_data[:, iterator*step:iterator*step+windows]
            EOG_datas_list.append(data_slice)
            EOG_labels_list.append(EOG_label)
    for trial_num in range(40):
        EMG_data = EMG_datas[trial_num]
        EMG_label = labels[trial_num]
        assert(EMG_data.shape == (256, 60))
        # 0-1化处理
        _max = EMG_data.max(axis=0).reshape(1, -1)
        _min = EMG_data.min(axis=0).reshape(1, -1)
        EMG_data = (EMG_data - _min) / (_max - _min)
        assert(EMG_data.shape == (256, 60))
        step = windows - overlapping
        iterator_num = int((60 - windows) / step + 1)
        for iterator in range(iterator_num):
            data_slice = EMG_data[:, iterator*step:iterator*step+windows]
            EMG_datas_list.append(data_slice)
            EMG_labels_list.append(EMG_label)
    for trial_num in range(40):
        GSR_data = GSR_datas[trial_num]
        GSR_label = labels[trial_num]
        assert(GSR_data.shape == (128, 60))
        # 0-1化处理
        _max = GSR_data.max(axis=0).reshape(1, -1)
        _min = GSR_data.min(axis=0).reshape(1, -1)
        GSR_data = (GSR_data - _min) / (_max - _min)
        assert(GSR_data.shape == (128, 60))
        step = windows - overlapping
        iterator_num = int((60 - windows) / step + 1)
        for iterator in range(iterator_num):
            data_slice = GSR_data[:, iterator*step:iterator*step+windows]
            GSR_datas_list.append(data_slice)
            GSR_labels_list.append(GSR_label)
    for trial_num in range(40):
        RSP_data = RSP_datas[trial_num]
        RSP_label = labels[trial_num]
        assert(RSP_data.shape == (128, 60))
        # 0-1化处理
        _max = RSP_data.max(axis=0).reshape(1, -1)
        _min = RSP_data.min(axis=0).reshape(1, -1)
        RSP_data = (RSP_data - _min) / (_max - _min)
        assert(RSP_data.shape == (128, 60))
        step = windows - overlapping
        iterator_num = int((60 - windows) / step + 1)
        for iterator in range(iterator_num):
            data_slice = RSP_data[:, iterator*step:iterator*step+windows]
            RSP_datas_list.append(data_slice)
            RSP_labels_list.append(RSP_label)
    for trial_num in range(40):
        BLV_data = BLV_datas[trial_num]
        BLV_label = labels[trial_num]
        assert(BLV_data.shape == (128, 60))
        # 0-1化处理
        _max = BLV_data.max(axis=0).reshape(1, -1)
        _min = BLV_data.min(axis=0).reshape(1, -1)
        BLV_data = (BLV_data - _min) / (_max - _min)
        assert(BLV_data.shape == (128, 60))
        step = windows - overlapping
        iterator_num = int((60 - windows) / step + 1)
        for iterator in range(iterator_num):
            data_slice = BLV_data[:, iterator*step:iterator*step+windows]
            BLV_datas_list.append(data_slice)
            BLV_labels_list.append(BLV_label)
    for trial_num in range(40):
        TMR_data = TMR_datas[trial_num]
        TMR_label = labels[trial_num]
        assert(TMR_data.shape == (128, 60))
        # 0-1化处理
        _max = TMR_data.max(axis=0).reshape(1, -1)
        _min = TMR_data.min(axis=0).reshape(1, -1)
        TMR_data = (TMR_data - _min) / (_max - _min+1e-8)
        assert(TMR_data.shape == (128, 60))
        step = windows - overlapping
        iterator_num = int((60 - windows) / step + 1)
        for iterator in range(iterator_num):
            data_slice = TMR_data[:, iterator*step:iterator*step+windows]
            TMR_datas_list.append(data_slice)
            TMR_labels_list.append(TMR_label)
    result = (EOG_datas_list, EOG_labels_list, EMG_datas_list, EMG_labels_list, GSR_datas_list, GSR_labels_list,
              RSP_datas_list, RSP_labels_list, BLV_datas_list, BLV_labels_list, TMR_datas_list, TMR_labels_list)
    return result
