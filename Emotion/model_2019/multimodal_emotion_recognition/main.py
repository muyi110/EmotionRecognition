#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common/") # 将其他模块路径添加到系统搜索路径
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from read_data import read_data as read_eeg_data
#from read_peripheral_data import read_data as read_per_data
from read_peripheral_raw_data import read_data as read_per_data
#from model import MMJLNNModel
from model_with_raw_data_input import MMJLNNModel

def split_datas_with_cross_validation(datas_all, labels, windows, seed=None):
    datas, EOG_datas, EMG_datas, GSR_datas, RSP_datas, BLV_datas, TMR_datas = datas_all
    samples_nums_one_trial = 60 - windows + 1
    assert(sum(labels[:samples_nums_one_trial]) == 0 or sum(labels[:samples_nums_one_trial]) == samples_nums_one_trial)
    datas_one = datas[list(labels==0)]
    datas_two = datas[list(labels==1)]
    EOG_datas_one = EOG_datas[list(labels==0)]
    EOG_datas_two = EOG_datas[list(labels==1)]
    EMG_datas_one = EMG_datas[list(labels==0)]
    EMG_datas_two = EMG_datas[list(labels==1)]
    GSR_datas_one = GSR_datas[list(labels==0)]
    GSR_datas_two = GSR_datas[list(labels==1)]
    RSP_datas_one = RSP_datas[list(labels==0)]
    RSP_datas_two = RSP_datas[list(labels==1)]
    BLV_datas_one = BLV_datas[list(labels==0)]
    BLV_datas_two = BLV_datas[list(labels==1)]
    TMR_datas_one = TMR_datas[list(labels==0)]
    TMR_datas_two = TMR_datas[list(labels==1)]
    labels_one = labels[list(labels==0)]
    labels_two = labels[list(labels==1)]
    assert(len(datas_one) // samples_nums_one_trial + len(datas_two) // samples_nums_one_trial == 40)

    label_one_number = (labels == 0).sum() # 获取类别 1 的数目
    label_two_number = (labels == 1).sum() # 获取类别 2 的数目   
    trial_label_one_number = int(label_one_number / samples_nums_one_trial) # 属于类别 1 的实验数
    trial_label_two_number = int(label_two_number / samples_nums_one_trial) # 属于类别 2 的实验数
    # 5 折交叉验证， 训练集有 32 个实验， 测试集有 8 个实验
    label_one_train_number = int(round(trial_label_one_number * 0.8))
    label_two_train_number = int(round(trial_label_two_number * 0.8))
    assert(label_one_train_number + label_two_train_number == 32)
    label_one_test_number = trial_label_one_number - label_one_train_number
    label_two_test_number = trial_label_two_number - label_two_train_number
    assert(label_one_test_number + label_two_test_number == 8)
    np.random.seed(seed)
    permutation_one = list(np.random.permutation(trial_label_one_number)) # 将数据随机打乱
    permutation_two = list(np.random.permutation(trial_label_two_number)) # 将数据随机打乱
    train_one_index = permutation_one[:label_one_train_number]
    train_two_index = permutation_two[:label_two_train_number]
    test_one_index = permutation_one[label_one_train_number:]
    test_two_index = permutation_two[label_two_train_number:]
    # 获取训练集和测试集
    train_datas = []
    train_labels = []
    train_datas_EOG = []
    train_datas_EMG = []
    train_datas_GSR = []
    train_datas_RSP = []
    train_datas_BLV = []
    train_datas_TMR = []
    for elem in train_one_index:
        train_datas.append(datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_EOG.append(EOG_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_EMG.append(EMG_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_GSR.append(GSR_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_RSP.append(RSP_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_BLV.append(BLV_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_TMR.append(TMR_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_labels.append(labels_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    for elem in train_two_index:
        train_datas.append(datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_EOG.append(EOG_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_EMG.append(EMG_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_GSR.append(GSR_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_RSP.append(RSP_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_BLV.append(BLV_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_datas_TMR.append(TMR_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_labels.append(labels_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    datas_train = np.r_[tuple(train_datas)]
    datas_train_EOG = np.r_[tuple(train_datas_EOG)]
    datas_train_EMG = np.r_[tuple(train_datas_EMG)]
    datas_train_GSR = np.r_[tuple(train_datas_GSR)]
    datas_train_RSP = np.r_[tuple(train_datas_RSP)]
    datas_train_BLV = np.r_[tuple(train_datas_BLV)]
    datas_train_TMR = np.r_[tuple(train_datas_TMR)]
    labels_train = np.c_[tuple(train_labels)].reshape(-1)
    assert(datas_train.shape == (32*samples_nums_one_trial, windows, 128))
    assert(datas_train_EOG.shape == (32*samples_nums_one_trial, windows, 256))
    assert(datas_train_EMG.shape == (32*samples_nums_one_trial, windows, 256))
    assert(datas_train_GSR.shape == (32*samples_nums_one_trial, windows, 128))
    assert(datas_train_RSP.shape == (32*samples_nums_one_trial, windows, 128))
    assert(datas_train_BLV.shape == (32*samples_nums_one_trial, windows, 128))
    assert(datas_train_TMR.shape == (32*samples_nums_one_trial, windows, 128))
    assert(labels_train.shape == (32*samples_nums_one_trial,))
    datas_train_all = (datas_train, datas_train_EOG, datas_train_EMG, datas_train_GSR, 
                       datas_train_RSP, datas_train_BLV, datas_train_TMR)
    test_datas = []
    test_datas_EOG = []
    test_datas_EMG = []
    test_datas_GSR = []
    test_datas_RSP = []
    test_datas_BLV = []
    test_datas_TMR = []
    test_labels = []
    for elem in test_one_index:
        test_datas.append(datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_EOG.append(EOG_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_EMG.append(EMG_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_GSR.append(GSR_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_RSP.append(RSP_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_BLV.append(BLV_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_TMR.append(TMR_datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_labels.append(labels_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    for elem in test_two_index:
        test_datas.append(datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_EOG.append(EOG_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_EMG.append(EMG_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_GSR.append(GSR_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_RSP.append(RSP_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_BLV.append(BLV_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_datas_TMR.append(TMR_datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_labels.append(labels_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    datas_test = np.r_[tuple(test_datas)]
    datas_test_EOG = np.r_[tuple(test_datas_EOG)]
    datas_test_EMG = np.r_[tuple(test_datas_EMG)]
    datas_test_GSR = np.r_[tuple(test_datas_GSR)]
    datas_test_RSP = np.r_[tuple(test_datas_RSP)]
    datas_test_BLV = np.r_[tuple(test_datas_BLV)]
    datas_test_TMR = np.r_[tuple(test_datas_TMR)]
    labels_test = np.c_[tuple(test_labels)].reshape(-1)
    assert(datas_test.shape == (8*samples_nums_one_trial, windows, 128))
    assert(datas_test_EOG.shape == (8*samples_nums_one_trial, windows, 256))
    assert(datas_test_EMG.shape == (8*samples_nums_one_trial, windows, 256))
    assert(datas_test_GSR.shape == (8*samples_nums_one_trial, windows, 128))
    assert(datas_test_RSP.shape == (8*samples_nums_one_trial, windows, 128))
    assert(datas_test_BLV.shape == (8*samples_nums_one_trial, windows, 128))
    assert(datas_test_TMR.shape == (8*samples_nums_one_trial, windows, 128))
    assert(labels_test.shape == (8*samples_nums_one_trial,))
    datas_test_all = (datas_test, datas_test_EOG, datas_test_EMG, datas_test_GSR, 
                       datas_test_RSP, datas_test_BLV, datas_test_TMR)
    return datas_train_all, labels_train, datas_test_all, labels_test

if __name__ == "__main__":
    people_num_list = list(range(2, 32))
    windows = 9               # 样本窗口大小
    accuracy_results_dic = {} # 一个字典，保存最终的结果
    F1_score_results_dic = {} # 一个字典，保存最终的结果
    samples_info_dic = {}
    for people_num_ in people_num_list:
        # 读EEG数据
        datas, labels = read_eeg_data(people_list=[people_num_], windows=windows, overlapping=windows-1, 
                                      classify_object_name=0, mv_flag=True, lds_flag=False)
        datas = np.array(datas)
        labels = np.array(labels)
        datas = datas.transpose((0,2,1))
        assert(datas.shape == ((60-windows+1)*40, windows, 128))
        assert(labels.shape == ((60-windows+1)*40,))
        # 读PER数据
        EOG_datas, EOG_labels, EMG_datas, EMG_labels, GSR_datas, GSR_labels, RSP_datas, RSP_labels,\
        BLV_datas, BLV_labels, TMR_datas, TMR_labels = read_per_data(people_list=[people_num_], windows=windows,
                                                                     overlapping=windows-1, classify_object_name=0)
        EOG_datas = np.array(EOG_datas).transpose((0, 2, 1))
        EOG_labels = np.array(EOG_labels)      
        EMG_datas = np.array(EMG_datas).transpose((0, 2, 1))
        EMG_labels = np.array(EMG_labels)      
        GSR_datas = np.array(GSR_datas).transpose((0, 2, 1))
        GSR_labels = np.array(GSR_labels)      
        RSP_datas = np.array(RSP_datas).transpose((0, 2, 1))
        RSP_labels = np.array(RSP_labels)      
        BLV_datas = np.array(BLV_datas).transpose((0, 2, 1))
        BLV_labels = np.array(BLV_labels)      
        TMR_datas = np.array(TMR_datas).transpose((0, 2, 1))
        TMR_labels = np.array(TMR_labels)      
        assert(EOG_datas.shape == ((60-windows+1)*40, windows, 256))
        assert(EMG_datas.shape == ((60-windows+1)*40, windows, 256))
        assert(GSR_datas.shape == ((60-windows+1)*40, windows, 128))
        assert(RSP_datas.shape == ((60-windows+1)*40, windows, 128))
        assert(BLV_datas.shape == ((60-windows+1)*40, windows, 128))
        assert(TMR_datas.shape == ((60-windows+1)*40, windows, 128))
        assert(sum(EOG_labels == labels) == 52*40)
        assert(sum(EMG_labels == labels) == 52*40)
        assert(sum(GSR_labels == labels) == 52*40)
        assert(sum(RSP_labels == labels) == 52*40)
        assert(sum(BLV_labels == labels) == 52*40)
        assert(sum(TMR_labels == labels) == 52*40)
        datas_all = (datas, EOG_datas, EMG_datas, GSR_datas, RSP_datas, BLV_datas, TMR_datas)

        seed_ = 1
        cross_validation_number = 5
        F1_scores_list = []
        accuracy_list = []
        samples_info = []
        for number in range(cross_validation_number):
            seed_ = seed_ + 1
            datas_train, train_labels, datas_test, test_labels = split_datas_with_cross_validation(datas_all, 
                                                                 labels, windows, seed_)
            print("train label 0: ", sum(train_labels==0), " train label 1: ", sum(train_labels==1))
            print("test label 0: ", sum(test_labels==0), " test label 1: ", sum(test_labels==1))
            train_label_0 = sum(train_labels==0)
            test_label_0 = sum(test_labels==0)
            label_0 = (train_label_0, test_label_0)
            samples_info.append(label_0)

            n_outputs = 2 # 二分类
            input_channels = datas_train[0].shape[-1] # 用于EEG数据
            seq_length = datas_train[0].shape[-2] # 序列的长度(EEG)
            dropout = 0.5
            num_channels = [128, 64, 32] # 有多少层，及每一层包含的神经元个数（这里的一层指一个 block）
            kernel_size = 3   # 卷积核大小  
            batch_size = 32*4

            # 开始构建 MMJLNN 模型实例
            mmjlnn = MMJLNNModel(sequence_length=seq_length, kernel_size=kernel_size, num_channels=num_channels,
                                 dropout=dropout, batch_size=batch_size, number_of_class_one=sum(train_labels==0)//52,
                                 in_channels=input_channels, random_state=42)
            mmjlnn.fit(X=datas_train, y=train_labels, epochs=351, X_test=datas_test, y_test=test_labels,
                     outputs=n_outputs, people_num=windows)

            mmjlnn.restore()
            y_pred = mmjlnn.predict(datas_test)
            total_acc_test = accuracy_score(test_labels, y_pred)
            y_pred_labels = list(y_pred)
            print("Test accuracy: {:.4f}%".format(total_acc_test*100))
            F1_scores_list.append(f1_score(test_labels, np.array(y_pred_labels)))
            temp = total_acc_test
            accuracy_list.append(temp)
        print("-------------------------------accuracy_list--------------------------------------")
        print(accuracy_list)
        print("accuacy mean : ", sum(accuracy_list) / 5)
        print("accuacy min: ", min(accuracy_list))
        print("accuacy max: ", max(accuracy_list))
        print("-------------------------------F1 score--------------------------------------")
        print(F1_scores_list)
        print("F1 score mean: ",sum(F1_scores_list)/5)
        print("F1 score min: ", min(F1_scores_list))
        print("F1 score max: ", max(F1_scores_list))
        print("-------------------------------sampels info--------------------------------------")
        print(samples_info)
        accuracy_results_dic[str(people_num_)] = accuracy_list + \
                                              [min(accuracy_list), max(accuracy_list), sum(accuracy_list)/5]
        F1_score_results_dic[str(people_num_)] = F1_scores_list + \
                                              [min(F1_scores_list), max(F1_scores_list), sum(F1_scores_list)/5]
        samples_info_dic[str(people_num_)] = samples_info
        print("accuracy: ")
        print(accuracy_results_dic)
        print("F1 score: ")
        print(F1_score_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/accuracy", accuracy_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/F1_score", F1_score_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/samples", samples_info_dic)
    print("accuracy: ")
    print(accuracy_results_dic)
    sum_ = 0
    for i in range(32):
        sum_ += accuracy_results_dic[str(i)][-1]
    print("acc: ", sum_/32)
    print("F1 score: ")
    print(F1_score_results_dic)
    sum_ = 0
    for i in range(32):
        sum_ += F1_score_results_dic[str(i)][-1]
    print("f1 acc: ", sum_/32)
    print("sample info")
    print(samples_info_dic)

