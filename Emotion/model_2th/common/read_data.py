# -*- coding:UTF-8 -*-
#################################################################################
# 此模块获取样本数据. ------2018-11-26
#################################################################################
import os
import math
import numpy as np
from features_extraction import differential_entropy, filter_with_stft
from features_smooth_mv import moving_average
from label_threshold_cluster import kmeans_cluster, get_samples_labels

SAMPLES_PATH = "../../samples/"

def index_generator(num_examples, batch_size, seed=0):
    '''
      此函数用于生成 batch 的索引
    '''
    np.random.seed(seed)
    permutation = list(np.random.permutation(num_examples))
    num_complete_minibatches = math.floor(num_examples/batch_size)
    for k in range(0, num_complete_minibatches):
        X_batch_index = permutation[k*batch_size:(k+1)*batch_size]
        y_batch_index = permutation[k*batch_size:(k+1)*batch_size]
        yield (X_batch_index, y_batch_index)
    if num_examples % batch_size != 0:
        X_batch_index = permutation[num_complete_minibatches*batch_size:num_examples]
        y_batch_index = permutation[num_complete_minibatches*batch_size:num_examples]
        yield (X_batch_index, y_batch_index)

def _get_samples_data(people_list, trial_list, path, windows=9, overlapping=8, classify_object_name=0):
    '''
      classify_object_name = 0 ----> valence
      classify_object_name = 1 ----> arousal
    '''
    labels_all_people = get_samples_labels(SAMPLES_PATH) # 读取 32 个人的样本标签数据
    valence_list, arousal_list, _, _ = kmeans_cluster(labels_all_people) # 获取每一个人的标签阈值
    labels_dic = {}
    labels_dic['0'] = valence_list
    labels_dic['1'] = arousal_list
    
    samples_dirs = os.listdir(path) # 目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    datas = [] # 获取最终的样本
    labels = [] # 对应的样本
    for people in people_list:
        for trial in trial_list:
            eeg = np.loadtxt(file_path[people]+"/trial_"+str(trial+1)+".csv", delimiter=',', 
                             skiprows=0, dtype=np.float32)
            eeg = eeg[:32, 128*3:] # 取后 60s 的 EEG 信号
            label_value = np.loadtxt(file_path[people]+"/label.csv", delimiter=",", skiprows=0,
                                     dtype=np.float32)[trial, :2]
            # label_value[0]-->valence, label_value[1]-->arousal
            if label_value[classify_object_name] >= labels_dic[str(classify_object_name)][people]:
                label = 0
            elif label_value[classify_object_name] < labels_dic[str(classify_object_name)][people]:
                label = 1
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            filtered_array = np.zeros((128, 128*60))
            for i in range(32):  # 依次处理 32 通道数据
                # 获取 4 个子频段 EEG 数据
                theta, alpha, beta, gamma = filter_with_stft(eeg[i])
                filtered_array[4*i] = theta
                filtered_array[4*i+1] = alpha
                filtered_array[4*i+2] = beta
                filtered_array[4*i+3] = gamma
            for iterator in range(iterator_num):
                data_slice = filtered_array[:, 128*(iterator*step):128*(iterator*step+windows)]
                datas.append(data_slice)
                labels.append(label)
    print("Get data success!")
    print("Total samples number is: ", len(labels))
    print("label 0: {}, label 1: {}".format(np.sum(np.array(labels)==0), np.sum(np.array(labels)==1)))
    return datas, labels

def read_data(people_list, trial_list, path=SAMPLES_PATH, windows=9, overlapping=8, 
              classify_object_name=0, train_flag=True, cross_validation_number=None):
    # datas 和 labels 都是 list. datas 中的每一项都是 shape=(128, 128*windows) 的 np.array
    datas, labels = _get_samples_data(people_list, trial_list, path, windows, overlapping, classify_object_name)
    datas_result = []
    for data in datas:
        data_list = []
        for window in range(windows):
            features_list = []
            for i in range(128):
                X = data[i, 128*window:128*(window+1)]
                if (i % 4) == 0:
                    features_list.append(differential_entropy(X, 4, 7))
                elif (i % 4) == 1:
                    features_list.append(differential_entropy(X, 8, 13))
                elif (i % 4) == 2:
                    features_list.append(differential_entropy(X, 14, 30))
                elif (i % 4) == 3:
                    features_list.append(differential_entropy(X, 31, 50))
            data_list.append(np.array(features_list).reshape(-1, 1))
        datas_result.append(np.c_[tuple(data_list)]) # 每一个元素 shape=(features, seq_length)=(128, 9)
    del datas
    assert len(datas_result) == len(labels)
    # 滑动平均滤波
    datas_result = moving_average(datas_result, windows, windows=20)
    _save_samples(datas_result, labels, people_list, train_flag, classify_object_name, cross_validation_number)
    return (datas_result, labels)

def _save_samples(datas_result, labels, people_list, train_flag, classify_object_name, cross_validation_number):
    if classify_object_name == 0:
        class_name = "valence"
    elif classify_object_name == 1:
        class_name = "arousal"
    if cross_validation_number == None:
        num = ""
    else:
        num = str(cross_validation_number)
    # 针对单独一个人情况
    if len(people_list) == 1:
        print("save samples start: ")
        if not os.path.isdir(os.path.join("../common/samples_single_people/"+class_name, "s"+str(people_list[0]))):
            os.makedirs(os.path.join("../common/samples_single_people/"+class_name, "s"+str(people_list[0])))
        if train_flag == True:
            np.save("../common/samples_single_people/"+class_name+"/s"+str(people_list[0])+"/train_datas"+num, 
                    datas_result)
            np.save("../common/samples_single_people/"+class_name+"/s"+str(people_list[0])+"/train_labels"+num, 
                    labels)
        elif train_flag == False:
            np.save("../common/samples_single_people/"+class_name+"/s"+str(people_list[0])+"/test_datas"+num, 
                    datas_result)
            np.save("../common/samples_single_people/"+class_name+"/s"+str(people_list[0])+"/test_labels"+num, 
                    labels)
        elif train_flag is None:
            np.save("./samples_single_people/"+class_name+"/s"+str(people_list[0])+"/datas"+num, 
                    datas_result)
            np.save("./samples_single_people/"+class_name+"/s"+str(people_list[0])+"/labels"+num, 
                    labels)
    if len(people_list) > 1:
        if train_flag == True:
            np.save("../common/samples_all_people/"+class_name+"/train_datas", datas_result)
            np.save("../common/samples_all_people/"+class_name+"/train_labels", labels)
        elif train_flag == False:
            np.save("../common/samples_all_people/"+class_name+"/test_datas", datas_result)
            np.save("../common/samples_all_people/"+class_name+"/test_labels", labels)
        elif train_flag is None:
            np.save("./samples_all_people/"+class_name+"/s"+str(people_list[0])+"/datas"+num, 
                    datas_result)
            np.save("./samples_all_people/"+class_name+"/s"+str(people_list[0])+"/labels"+num, 
                    labels)

if __name__ == "__main__":
    trial_list = list(range(0, 40))
    for number in range(32):
        people_list = [number]
        datas, labels = read_data(people_list, trial_list, path=SAMPLES_PATH, windows=9, overlapping=8, 
                                  classify_object_name=0, train_flag=None, cross_validation_number=None)
        assert(len(datas) == 40*52)
        assert(len(labels) == 40*52)
    for number in range(32):
        people_list = [number]
        datas, labels = read_data(people_list, trial_list, path=SAMPLES_PATH, windows=9, overlapping=8, 
                                  classify_object_name=1, train_flag=None, cross_validation_number=None)
        assert(len(datas) == 40*52)
        assert(len(labels) == 40*52)
