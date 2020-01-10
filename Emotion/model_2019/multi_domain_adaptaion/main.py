#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common/") # 将其他模块路径添加到系统搜索路径
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from read_data import read_data
from model import MDANN
from sklearn.manifold import TSNE # 用于高维特征可视化
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score
from utils import plot_embedding

def split_datas_with_cross_validation(datas, labels, windows, seed=None):
    samples_nums_one_trial = 60 - windows + 1
    assert(datas.shape == ((60-windows+1)*40, windows, 128)) # 128->60
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
    assert(datas_train.shape == (32*samples_nums_one_trial, windows, 128)) # 128->60
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
    assert(datas_test.shape == (8*samples_nums_one_trial, windows, 128)) # 128->60
    assert(labels_test.shape == (8*samples_nums_one_trial,))
    return datas_train, labels_train, datas_test, labels_test

if __name__ == "__main__":
    people_num_list = list(range(0, 32))
    windows = 9               # 样本窗口大小
    accuracy_results_dic = {} # 一个字典，保存最终的结果
    F1_score_results_dic = {} # 一个字典，保存最终的结果
    samples_info_dic = {}
    silhouette_score_dic = {}
    normalized_mutual_info_score_dic = {}
    adjusted_mutual_info_score_dic = {}
    for people_num_ in people_num_list:
        datas, labels = read_data(people_list=[people_num_], windows=windows, overlapping=windows-1, 
                                  classify_object_name=0, mv_flag=True, lds_flag=False)
        datas = np.array(datas)
        labels = np.array(labels)
        #channels = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31, 36, 37, 38, 39, 56, 57, 58, 59,80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127] # 探索不同的通道
        #datas = datas[:, channels, :]
        #print(datas.shape)
        datas = datas.transpose((0,2,1))
        assert(datas.shape == ((60-windows+1)*40, windows, 128)) # 128->60
        assert(labels.shape == ((60-windows+1)*40,))
        seed_ = 1
        cross_validation_number = 5
        F1_scores_list = []
        accuracy_list = []
        samples_info = []
        silhouette_score_list = []
        normalized_mutual_info_score_list = []
        adjusted_mutual_info_score_list = []
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
            batch_size = 32*4

            # 开始构建 MDANN 模型实例
            mdann = MDANN(sequence_length=seq_length, kernel_size=kernel_size, num_channels=num_channels,
                          dropout=dropout, batch_size=batch_size, number_of_class_one=sum(train_labels==0)//52,
                          in_channels=input_channels, random_state=42)
            mdann.fit(X=datas_train, y=train_labels, epochs=351, X_test=datas_test, y_test=test_labels,
                     outputs=n_outputs, people_num=windows)

            mdann.restore()
            y_pred = mdann.predict(datas_test)
            total_acc_test = accuracy_score(test_labels, y_pred)
            y_pred_labels = list(y_pred)
            print("Test accuracy: {:.4f}%".format(total_acc_test*100))
            F1_scores_list.append(f1_score(test_labels, np.array(y_pred_labels)))
            temp = total_acc_test
            accuracy_list.append(temp)
            # 下面计算类别结果评价指标
            normalized_mutual_info_score_list.append(normalized_mutual_info_score(test_labels, y_pred.reshape(-1)))
            adjusted_mutual_info_score_list.append(adjusted_mutual_info_score(test_labels, y_pred.reshape(-1)))
            if(len(np.unique(y_pred.reshape(-1))) == 2):
                silhouette_score_list.append(silhouette_score(mdann._session.run(mdann.feature, feed_dict={mdann.inputs:datas_test}), y_pred.reshape(-1)))
            else:
                silhouette_score_list.append(0)
            # 下面是特征可视化
            #index = list(np.random.permutation(1664))[:]
            #combined_test_data = np.vstack([datas_train[index], datas_test[:]])
            #combined_test_labels = np.hstack([train_labels[index], test_labels[:]])
            #combined_test_domain = np.hstack([np.tile([0], [1664]), np.tile([1], [416])])

            #test_emd = mdann._session.run(mdann.feature, feed_dict={mdann.inputs:combined_test_data})
            #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
            #mdann_tsne = tsne.fit_transform(test_emd)
            #plot_embedding(mdann_tsne, combined_test_labels, combined_test_domain, people_num_, number, './result/fig_V/') 
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
        print("-------------------------------metrics--------------------------------------")
        print("adjusted_mutual_info_score: ", adjusted_mutual_info_score_list)
        print("adjusted_mutual_info_score mean: ", sum(adjusted_mutual_info_score_list) / 5)
        print("normalized_mutual_info_score: ", normalized_mutual_info_score_list)
        print("normalized_mutual_info_score mean: ", sum(normalized_mutual_info_score_list) / 5)
        print("silhouette_score: ", silhouette_score_list)
        print("silhouette_score mean: ", sum(silhouette_score_list) / 5)
        accuracy_results_dic[str(people_num_)] = accuracy_list + \
                                              [min(accuracy_list), max(accuracy_list), sum(accuracy_list)/5]
        F1_score_results_dic[str(people_num_)] = F1_scores_list + \
                                              [min(F1_scores_list), max(F1_scores_list), sum(F1_scores_list)/5]
        samples_info_dic[str(people_num_)] = samples_info
        adjusted_mutual_info_score_dic[str(people_num_)] = adjusted_mutual_info_score_list + [sum(adjusted_mutual_info_score_list) / 5]
        normalized_mutual_info_score_dic[str(people_num_)] = normalized_mutual_info_score_list + [sum(normalized_mutual_info_score_list) / 5]
        silhouette_score_dic[str(people_num_)] = silhouette_score_list + [sum(silhouette_score_list) / 5]
        print("accuracy: ")
        print(accuracy_results_dic)
        print("F1 score: ")
        print(F1_score_results_dic)
        print("adjusted_mutual_info_score: ")
        print(adjusted_mutual_info_score_dic)
        print("normalized_mutual_info_score:")
        print(normalized_mutual_info_score_dic)
        print("silhouette_score")
        print(silhouette_score_dic)
    np.save("./result/mv/valence/"+str(windows)+"/accuracy_0804_128", accuracy_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/F1_score_0804_128", F1_score_results_dic)
    np.save("./result/mv/valence/"+str(windows)+"/samples_0804_128", samples_info_dic)
    np.save("./result/mv/valence/"+str(windows)+"/adjusted_mutual_info_score_0804_128", adjusted_mutual_info_score_dic)
    np.save("./result/mv/valence/"+str(windows)+"/normalized_mutual_info_score_dic_0804_128", normalized_mutual_info_score_dic)
    np.save("./result/mv/valence/"+str(windows)+"/silhouette_score_dic_0804_128", silhouette_score_dic)
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
    print("metrics: ")
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(32):
        sum_1 += adjusted_mutual_info_score_dic[str(i)][-1]
        sum_2 += normalized_mutual_info_score_dic[str(i)][-1]
        sum_3 += silhouette_score_dic[str(i)][-1]
    print("adjusted_mutual_info_score: ", sum_1 / 32)
    print("normalized_mutual_info_score: ", sum_2 / 32)
    print("silhouette_score: ", sum_3 / 32)

