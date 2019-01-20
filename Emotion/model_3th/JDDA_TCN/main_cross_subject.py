#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common/")
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from read_data import read_data
from sklearn.manifold import TSNE
from utils import plot_embedding
from jdda import JDDA_Model

def split_samples():
    '''
    leave-one-subject cross validation.
    '''
    train_test_people_numbers_list = []
    total_samples_list = list(range(32))
    for i in range(32):
        total_samples_list_temp = total_samples_list.copy()
        test_number_list = [i]
        total_samples_list_temp.remove(i)
        train_numbers_list = total_samples_list_temp
        train_test_numbers = (train_numbers_list, test_number_list)
        train_test_people_numbers_list.append(train_test_numbers)
    return train_test_people_numbers_list

if __name__ == "__main__":
    train_test_people_numbers_list = split_samples()
    windows = 9
    accuracy_results_dic = {}
    F1_score_results_dic = {}
    F1_scores_list = []
    accuracy_list = []
    for train_test_people in train_test_people_numbers_list:
        train_people, test_people = train_test_people
        print(train_people)
        datas_train, train_labels = read_data(people_list=train_people, windows=windows, overlapping=windows-1,
                                              classify_object_name=0, mv_flag=True, lds_flag=False)
        datas_test, test_labels = read_data(people_list=test_people, windows=windows, overlapping=windows-1,
                                            classify_object_name=0, mv_flag=True, lds_flag=False)
        datas_train = np.array(datas_train).transpose((0,2,1))
        train_labels = np.array(train_labels)
        datas_test = np.array(datas_test).transpose((0,2,1))
        test_labels = np.array(test_labels)
        assert(datas_train.shape==((60-windows+1)*40*31, windows, 128))
        assert(datas_test.shape==((60-windows+1)*40*1, windows, 128))
        print("train label 0: ", sum(train_labels==0), "train label 1: ", sum(train_labels==1))
        print("test label 0: ", sum(test_labels==0), "test label 1: ", sum(test_labels==1))
        n_outputs = 2
        input_channels = datas_train.shape[-1]
        seq_length = datas_train.shape[-2]
        dropout = 0.5
        num_channels = [128, 64, 32]
        kernel_size = 3
        batch_size = 256

        model = JDDA_Model(sequence_length=seq_length, kernel_size=kernel_size, num_channels=num_channels, 
                           dropout=dropout, batch_size=batch_size, in_channels=input_channels, random_state=42)
        model.fit(X=datas_train, y=train_labels, epochs=451, X_test=datas_test, y_test=test_labels, 
                  outputs=n_outputs, people_num=windows)
        model.restore()
        y_pred = model.predict(datas_test)
        total_acc_test = accuracy_score(test_labels, y_pred)
        y_pred_labels = list(y_pred)
        print("Test accuracy: {:.4f}%".format(total_acc_test*100))
        F1_scores_list.append(f1_score(test_labels, np.array(y_pred_labels)))
        temp = total_acc_test
        accuracy_list.append(temp)
        print("-------------------------------------accuracy list-------------------------------------------")
        print(accuracy_list)
        print("-------------------------------------F1 score------------------------------------------------")
        print(F1_scores_list)
