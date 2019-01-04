#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common/") # 将其他模块路径添加到系统搜索路径
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from read_data import read_data
# from read_raw_data import read_data
from mdann import MDANNModel
from sklearn.manifold import TSNE # 用于高维特征可视化
from utils import plot_embedding

def split_datas_with_cross_validation(datas, labels, windows, seed=None):
    samples_nums_one_trial = 60 - windows + 1
    assert(datas.shape == ((60-windows+1)*40, windows, 128))
    assert(labels.shape == ((60-windows+1)*40,))
    assert(sum(labels[:samples_nums_one_trial]) == 0 or sum(labels[:samples_nums_one_trial]) == samples_nums_one_trial)
    datas_one = datas[list(labels==0)]
    datas_two = datas[list(labels==1)]
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
    for elem in train_one_index:
        train_datas.append(datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_labels.append(labels_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    for elem in train_two_index:
        train_datas.append(datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_labels.append(labels_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    datas_train = np.r_[tuple(train_datas)]
    labels_train = np.c_[tuple(train_labels)].reshape(-1)
    assert(datas_train.shape == (32*samples_nums_one_trial, windows, 128))
    assert(labels_train.shape == (32*samples_nums_one_trial,))
    test_datas = []
    test_labels = []
    for elem in test_one_index:
        test_datas.append(datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_labels.append(labels_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    for elem in test_two_index:
        test_datas.append(datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_labels.append(labels_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    datas_test = np.r_[tuple(test_datas)]
    labels_test = np.c_[tuple(test_labels)].reshape(-1)
    assert(datas_test.shape == (8*samples_nums_one_trial, windows, 128))
    assert(labels_test.shape == (8*samples_nums_one_trial,))
    return datas_train, labels_train, datas_test, labels_test

if __name__ == "__main__":
    people_num_list = list(range(0, 32))
    windows = 9               # 样本窗口大小
    accuracy_results_dic = {} # 一个字典，保存最终的结果
    F1_score_results_dic = {} # 一个字典，保存最终的结果
    samples_info_dic = {}
    for people_num_ in people_num_list:
        datas, labels = read_data(people_list=[people_num_], windows=windows, overlapping=windows-1, 
                                  classify_object_name=0, mv_flag=True, lds_flag=False)
        # datas, labels = read_data(people_list=[people_num_], trial_list=list(range(40)), path="../../samples/"
        #                           , classify_object_name=0)
        datas = np.array(datas)
        labels = np.array(labels)
        datas = datas.transpose((0,2,1))
        assert(datas.shape == ((60-windows+1)*40, windows, 128))
        assert(labels.shape == ((60-windows+1)*40,))
        seed_ = 1
        cross_validation_number = 5
        F1_scores_list = []
        accuracy_list = []
        samples_info = []
        for number in range(cross_validation_number):
            seed_ = seed_ + 1
            datas_train, train_labels, datas_test, test_labels = split_datas_with_cross_validation(datas, 
                                                                 labels, windows, seed_)
            print("train label 0: ", sum(train_labels==0), " train label 1: ", sum(train_labels==1))
            print("test label 0: ", sum(test_labels==0), " test label 1: ", sum(test_labels==1))
            train_label_0 = sum(train_labels==0)
            test_label_0 = sum(test_labels==0)
            label_0 = (train_label_0, test_label_0)
            samples_info.append(label_0)

            n_outputs = 2 # 二分类
            input_channels = datas_train.shape[-1]
            seq_length = datas_train.shape[-2] # 序列的长度
            dropout = 0.5
            num_channels = [128, 64, 32] # 有多少层，及每一层包含的神经元个数（这里的一层指一个 block）
            kernel_size = 3   # 卷积核大小  
            batch_size = 4*32

            # 开始构建 DANN 模型实例
            dann = MDANNModel(sequence_length=seq_length, kernel_size=kernel_size, num_channels=num_channels,
                             dropout=dropout, batch_size=batch_size, in_channels=input_channels, train_ratio=0.5,
                             random_state=42)
            dann.fit(X=datas_train, y=train_labels, num_steps=205001, X_test=datas_test, y_test=test_labels,
                     outputs=n_outputs, people_num=windows, training_mode="dann")

            dann.restore()
            y_pred = dann.predict(datas_test)
            total_acc_test = accuracy_score(test_labels, y_pred)
            y_pred_labels = list(y_pred)
            print("Test accuracy: {:.4f}%".format(total_acc_test*100))
            F1_scores_list.append(f1_score(test_labels, np.array(y_pred_labels)))
            temp = total_acc_test
            accuracy_list.append(temp)
            # 下面是特征可视化
            combined_test_data = np.vstack([datas_train[:100], datas_test[:100]])
            combined_test_labels = np.hstack([train_labels[:100], test_labels[:100]])
            combined_test_domain = np.hstack([np.tile([0], [100]), np.tile([1], [100])])

            test_emd = dann._session.run(dann.feature, feed_dict={dann.inputs:combined_test_data})
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
            dann_tsne = tsne.fit_transform(test_emd)
            plot_embedding(dann_tsne, combined_test_labels, combined_test_domain) 
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
    np.save("./result/mv/valence/"+str(windows)+"/accuracy", accuracy_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/F1_score", F1_score_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/samples", samples_info_dic)
    print("accuracy: ")
    print(accuracy_results_dic)
    print("F1 score: ")
    print(F1_score_results_dic)
    print("sample info")
    print(samples_info_dic)

