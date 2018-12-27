# -*- coding:UTF-8 -*-
import os
import math
import numpy as np

SAMPLES_PATH = '../../data_analysis/samples/'''

def get_samples_data(path, train=True, test=False, seed=42):
    samples_dirs = os.listdir(path) # 目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    datas = []
    labels = []
    np.random.seed(42)
    samples_indices = list(np.random.permutation(32*40)) # 将所有的样本随机打乱
    # 将样本划分为训练集和测试集（训练集：880个;  测试集：400个）
    train_indices = samples_indices[:880]
    test_indices = samples_indices[880:]
    # 获取训练集样本
    if train:
        for elem in train_indices:
            people = elem // 40 # 获取该样本实验属于哪个人
            trail_num = elem % 40 # 获取是第几个实验
            data = np.loadtxt(file_path[people]+'/trial_'+str(trail_num + 1)+".csv", delimiter=',',
                              skiprows=0, dtype=np.float32)
            datas.append(data)
            # 获取对应的 labels
            labels_value = np.loadtxt(file_path[people]+'/label.csv', delimiter=',', 
                                      skiprows=0, dtype=np.float32)[trail_num,:2]
            if labels_value[0] > 5. and labels_value[1] > 5.:
                label = 1 # 第一象限
            elif labels_value[0] >= 5. and labels_value[1] <= 5.:
                label = 2 # 第二象限
            elif labels_value[0] < 5. and labels_value[1] <= 5.:
                label = 3 # 第三象限
            elif labels_value[0] <= 5. and labels_value[1] > 5.:
                label = 4 # 第四象限
            labels.append(label)
        print("Get train data success!")
    # 获取测试集样本
    elif test:
        for elem in test_indices:
            people = elem // 40 # 获取该样本实验属于哪个人
            trail_num = elem % 40 # 获取是第几个实验
            data = np.loadtxt(file_path[people]+'/trial_'+str(trail_num + 1)+".csv", delimiter=',',
                              skiprows=0, dtype=np.float32)
            datas.append(data)
            # 获取对应的 labels
            labels_value = np.loadtxt(file_path[people]+'/label.csv', delimiter=',', 
                                      skiprows=0, dtype=np.float32)[trail_num,:2]
            if labels_value[0] > 5. and labels_value[1] > 5.:
                label = 1 # 第一象限
            elif labels_value[0] >= 5. and labels_value[1] <= 5.:
                label = 2 # 第二象限
            elif labels_value[0] < 5. and labels_value[1] <= 5.:
                label = 3 # 第三象限
            elif labels_value[0] <= 5. and labels_value[1] > 5.:
                label = 4 # 第四象限
            labels.append(label)
        print("Get test data success!")
    # 读取全部的样本
    else:
        for people in range(32):
            for trial in range(40):
                data = np.loadtxt(file_path[people]+'/trial_'+str(trial+1)+".csv", delimiter=',', 
                                  skiprows=0, dtype=np.float32)
                datas.append(data)
                # 获取对应的 labels
                labels_value = np.loadtxt(file_path[people]+'/label.csv', delimiter=',', 
                                          skiprows=0, dtype=np.float32)[trail,:2]
                if labels_value[0] > 5. and labels_value[1] > 5.:
                    label = 1 # 第一象限
                elif labels_value[0] >= 5. and labels_value[1] <= 5.:
                    label = 2 # 第二象限
                elif labels_value[0] < 5. and labels_value[1] <= 5.:
                    label = 3 # 第三象限
                elif labels_value[0] <= 5. and labels_value[1] > 5.:
                    label = 4 # 第四象限
                labels.append(label)
        print("Get all data success!")
    return (datas, labels)

def index_generator(num_examples, batch_size, seed=0):
    '''此函数用于生成 batch 的索引'''
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

def read_data(path=SAMPLES_PATH, train=True, test=False, seed=42, input_datas_norm=True):
    '''默认读取训练集数据（920个）'''
    # datas 和 labels 都是 list. datas 中的每一项是 shape=(40, 8064) 的数组
    datas, labels = get_samples_data(path, train=train, test=test, seed=42)
    # 移除前 3S 的数据，利用后 60S 的数据
    # 同时将数据归一化处理
    datas_result = []
    for data in datas:
        if input_datas_norm:
            mean = data[:,128*3:].mean()
            std = data[:,128*3:].std()
            norm_result = (data[:,128*3:] - mean) / (std+1e-10)
            datas_result.append(norm_result)
        else:
            datas_result.append(data[:,128*3:])
    return (datas_result, labels)


if __name__ == "__main__":
    datas, labels = read_data(SAMPLES_PATH, train=False, test=True, input_datas_norm=False)
    print(len(datas))
    print(labels)
    print(datas[0][0,:128])
    print(datas[5].shape)
