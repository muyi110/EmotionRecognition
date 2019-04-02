# -*- coding:UTF-8 -*-
import numpy as np

def batch_generator(X, y, batch_size, seed):
    EEG_X, EOG_X, EMG_X, GSR_X, RSP_X, BLV_X, TMR_X = X
    assert(EEG_X.shape[0] == 52*32)
    assert(EOG_X.shape[0] == 52*32)
    assert(EMG_X.shape[0] == 52*32)
    assert(GSR_X.shape[0] == 52*32)
    assert(RSP_X.shape[0] == 52*32)
    assert(BLV_X.shape[0] == 52*32)
    assert(TMR_X.shape[0] == 52*32)
    assert(y.shape == (52*32,))
    assert(np.sum(y[:52]) == 0 or np.sum(y[:52]) == 52)
    X_class_one = EEG_X[y==0]
    X_class_two = EEG_X[y==1]
    EOG_class_one = EOG_X[y==0]
    EOG_class_two = EOG_X[y==1]
    EMG_class_one = EMG_X[y==0]
    EMG_class_two = EMG_X[y==1]
    GSR_class_one = GSR_X[y==0]
    GSR_class_two = GSR_X[y==1]
    RSP_class_one = RSP_X[y==0]
    RSP_class_two = RSP_X[y==1]
    BLV_class_one = BLV_X[y==0]
    BLV_class_two = BLV_X[y==1]
    TMR_class_one = TMR_X[y==0]
    TMR_class_two = TMR_X[y==1]
    y_class_one = y[y==0]
    y_class_two = y[y==1]
    assert(X_class_one.shape[0] + X_class_two.shape[0] == 32*52)
    assert(X_class_one.shape == (np.sum(y==0), 9, 128))
    np.random.seed(seed)
    permutation = list(np.random.permutation(52))
    
    num_complete_minibatches = 52 // (batch_size//32)
    minibatchs = []
    for k in range(num_complete_minibatches):
        X_s = permutation[k*(batch_size//32):(k+1)*(batch_size//32)]
        y_s = permutation[k*(batch_size//32):(k+1)*(batch_size//32)]
        temp_list_X_one = []
        temp_list_EOG_X_one = []
        temp_list_EMG_X_one = []
        temp_list_GSR_X_one = []
        temp_list_RSP_X_one = []
        temp_list_BLV_X_one = []
        temp_list_TMR_X_one = []
        temp_list_y_one = []
        for i in range(X_class_one.shape[0]//52):
            temp_list_X_one.append(X_class_one[list(np.array(X_s)+52*i)])
            temp_list_EOG_X_one.append(EOG_class_one[list(np.array(X_s)+52*i)])
            temp_list_EMG_X_one.append(EMG_class_one[list(np.array(X_s)+52*i)])
            temp_list_GSR_X_one.append(GSR_class_one[list(np.array(X_s)+52*i)])
            temp_list_RSP_X_one.append(RSP_class_one[list(np.array(X_s)+52*i)])
            temp_list_BLV_X_one.append(BLV_class_one[list(np.array(X_s)+52*i)])
            temp_list_TMR_X_one.append(TMR_class_one[list(np.array(X_s)+52*i)])
            temp_list_y_one.append(y_class_one[list(np.array(y_s)+52*i)])
        class_one_X = np.vstack(tuple(temp_list_X_one))
        class_one_EOG_X = np.vstack(tuple(temp_list_EOG_X_one))
        class_one_EMG_X = np.vstack(tuple(temp_list_EMG_X_one))
        class_one_GSR_X = np.vstack(tuple(temp_list_GSR_X_one))
        class_one_RSP_X = np.vstack(tuple(temp_list_RSP_X_one))
        class_one_BLV_X = np.vstack(tuple(temp_list_BLV_X_one))
        class_one_TMR_X = np.vstack(tuple(temp_list_TMR_X_one))
        class_one_y = np.hstack(tuple(temp_list_y_one))
        temp_list_X_two = []
        temp_list_EOG_X_two = []
        temp_list_EMG_X_two = []
        temp_list_GSR_X_two = []
        temp_list_RSP_X_two = []
        temp_list_BLV_X_two = []
        temp_list_TMR_X_two = []
        temp_list_y_two = []
        for i in range(X_class_two.shape[0]//52):
            temp_list_X_two.append(X_class_two[list(np.array(X_s)+52*i)])
            temp_list_EOG_X_two.append(EOG_class_two[list(np.array(X_s)+52*i)])
            temp_list_EMG_X_two.append(EMG_class_two[list(np.array(X_s)+52*i)])
            temp_list_GSR_X_two.append(GSR_class_two[list(np.array(X_s)+52*i)])
            temp_list_RSP_X_two.append(RSP_class_two[list(np.array(X_s)+52*i)])
            temp_list_BLV_X_two.append(BLV_class_two[list(np.array(X_s)+52*i)])
            temp_list_TMR_X_two.append(TMR_class_two[list(np.array(X_s)+52*i)])
            temp_list_y_two.append(y_class_two[list(np.array(y_s)+52*i)])
        class_two_X = np.vstack(tuple(temp_list_X_two))
        class_two_EOG_X = np.vstack(tuple(temp_list_EOG_X_two))
        class_two_EMG_X = np.vstack(tuple(temp_list_EMG_X_two))
        class_two_GSR_X = np.vstack(tuple(temp_list_GSR_X_two))
        class_two_RSP_X = np.vstack(tuple(temp_list_RSP_X_two))
        class_two_BLV_X = np.vstack(tuple(temp_list_BLV_X_two))
        class_two_TMR_X = np.vstack(tuple(temp_list_TMR_X_two))
        class_two_y = np.hstack(tuple(temp_list_y_two))
        EEG_X_batch = np.vstack([class_one_X, class_two_X])
        EOG_X_batch = np.vstack([class_one_EOG_X, class_two_EOG_X]) 
        EMG_X_batch = np.vstack([class_one_EMG_X, class_two_EMG_X]) 
        GSR_X_batch = np.vstack([class_one_GSR_X, class_two_GSR_X]) 
        RSP_X_batch = np.vstack([class_one_RSP_X, class_two_RSP_X]) 
        BLV_X_batch = np.vstack([class_one_BLV_X, class_two_BLV_X]) 
        TMR_X_batch = np.vstack([class_one_TMR_X, class_two_TMR_X]) 
        y_batch = np.hstack([class_one_y, class_two_y])
        mnibatch = (EEG_X_batch, EOG_X_batch, EMG_X_batch, GSR_X_batch, RSP_X_batch, BLV_X_batch, TMR_X_batch, y_batch)
        assert(EEG_X_batch.shape == (batch_size, 9, 128))
        assert(EOG_X_batch.shape == (batch_size, 9, 256))
        assert(EMG_X_batch.shape == (batch_size, 9, 256))
        assert(GSR_X_batch.shape == (batch_size, 9, 128))
        assert(RSP_X_batch.shape == (batch_size, 9, 128))
        assert(BLV_X_batch.shape == (batch_size, 9, 128))
        assert(TMR_X_batch.shape == (batch_size, 9, 128))
        assert(y_batch.shape == (batch_size,))
        minibatchs.append(mnibatch)
    return minibatchs, X_class_one.shape[0]//52
