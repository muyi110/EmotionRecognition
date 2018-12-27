# -*- coding=UTF-8 -*-
################################################################################
# 对数据或特征的平滑处理
# 线性动态系统 (LDS)
################################################################################
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def linear_dynamical_systems(X, seq_length, model):
    '''
      对数据做 LDS 平均处理
      X: shape=(samples, features, seq_length)
    '''
    # 构建 LDS 实例
    lds = LDS(d=128, k=2, model=model)

    X = np.array(X)
    numbers_one_trial = 60 - seq_length + 1 # 一个实验划分的样本数
    number_samples = X.shape[0]
    assert (number_samples % numbers_one_trial == 0)
    numbers_trial = number_samples // numbers_one_trial # 获取共有多少个实验(一个实验是 60s )
    trial_list = []
    result_list = []
    for trial in range(numbers_trial):
        trial_list.append(X[trial*numbers_one_trial:(trial+1)*numbers_one_trial,:,:])
    # 开始对每一个实验进行 LDS 平滑处理
    one_trial_features = None
    for trial in trial_list:
        # 每一个 trial 的 shape=(numbers_one_trial, features, seq_length)
        # 开始将 trial 拼接为 60s 长的 shape=(features, 60)
        temp = []
        for i in range(1, numbers_one_trial):
            temp.append(trial[i, :, -1])
        if len(temp) != 0:
            one_trial_features = np.c_[tuple(temp)]
            one_trial_features = np.c_[trial[0, :, :], one_trial_features]
        else:
            one_trial_features = np.c_[trial[0,:,:]]
        assert (one_trial_features.shape == (128, 60))
        # 开始利用 LDS 进行特征平滑处理（沿时间轴）
        temp_X = one_trial_features.copy()
        _, one_trial_features = lds.kalman_filter(temp_X, model)
        # temp_value = one_trial_features.copy()
        # for step in range(one_trial_features.shape[1]):
        #     one_trial_features[:, step:step+1] = samples_gauss(mean=temp_value[:, step:step+1], cov=model['S'])
        assert (one_trial_features.shape == (128, 60))
        # 将滤波后的 60s 长按照原来时间窗口，还原
        for i in range(numbers_one_trial):
            result_list.append(one_trial_features[:, i:(i+seq_length)])

    assert (np.array(result_list).shape == X.shape)
    return result_list

def samples_gauss(mean, cov):
    mean = mean.reshape(-1)
    assert(cov.shape == (mean.shape[0], mean.shape[0]))
    sample = np.random.multivariate_normal(mean=mean, cov=cov)
    assert(sample.shape == mean.shape)
    sample = sample.reshape(-1, 1)
    return sample

class LDS():
    '''
      线性动态系统实现
    '''
    def __init__(self, d, k, model=None):
        '''
          model: 一个字典，存储 LDS 的模型参数
          d: 观测变量的维数
          k: 隐变量维数
        '''
        if model:
            self._A = model['A']
            self._G = model['G']
            self._C = model['C']
            self._S = model['S']
            self._mu0 = model['mu0']
            self._P0 = model['P0']
        else:
            self._A = np.random.randn(k, k)
            self._G = scipy.stats.invwishart(df=k, scale=np.eye(k)).rvs(size=1)
            self._C = np.random.randn(d, k)
            self._S = scipy.stats.invwishart(df=d, scale=np.eye(d)).rvs(size=1)
            self._mu0 = np.random.randn(k, 1)
            self._P0 = scipy.stats.invwishart(df=k, scale=np.eye(k)).rvs(size=1)
    
    def kalman_filter(self, X, model):
        '''
          X: 输入观测序列 shape=(features, seq_length) = (128, 60)
        '''
        A = model['A']
        G = model['G']
        C = model['C']
        S = model['S']
        mu0 = model['mu0']
        P0 = model['P0']
        n = X.shape[1]          # 获取序列的长度 (60)
        d = X.shape[0]          # 观测变量的维度
        k = mu0.shape[0]        # 获取隐变量的维数
        mu = np.zeros((k, n))   # 均值
        V = np.zeros((k, k, n)) # 方差（协方差）
        llh = np.zeros((1, n))  # 对数似然函数
        c = np.zeros((d, n))
        I = np.eye(k)

        P0C = np.matmul(P0, C.T)
        R = np.matmul(C, P0C) + S
        K = np.matmul(P0C, np.linalg.inv(R))                                 # PRML 公式 13.97
        mu[:, 0:1] = mu0 + np.matmul(K, X[:, 0:1] - np.matmul(C, mu0))       # PRML 公式 13.94
        V[:, :, 0] = np.matmul((I - np.matmul(K, C)), P0)                    # PRML 公式 13.95
        c[:, 0:1] = np.matmul(C, mu[:,0:1])                                        # PRML 公式 13.96 中均值
        llh[0, 0] = self._logGauss(X[:, 0:1], np.matmul(C, mu0), R)
        for i in range(1, n):
            mu[:, i:i+1], V[:, :, i], c[:, i:i+1], llh[0, i] = self._forward_step(X[:, i].reshape(-1, 1), mu[:, i-1:i],
                                                                                  V[:, :, i-1], A, G, C, S, I)
        llh = llh.sum()
        # llh 表示观测序列似然函数，c 可以认为是观测序列经过 LDS 滤波后的序列
        return llh, c

    def _logGauss(self, X, mu, sigma):
        d = X.shape[0]                                             # 获取观测变量的维数
        X = X - mu
        U = np.linalg.cholesky(sigma).T                            # 获取 sigma 的上三角矩阵
        Q = np.matmul(np.linalg.inv(U.T), X)
        q = (Q * Q).sum(axis=0)                                    # 计算二项式项
        c = d * np.log(2*np.pi) + (2 * np.log(U.diagonal()).sum()) # 计算常数项
        y = -(c + q) / 2
        return y

    def _forward_step(self, x, mu, V, A, G, C, S, I):
        P = np.matmul(np.matmul(A, V), A.T) + G  # PRML 公式 13.88
        PC = np.matmul(P, C.T)
        R = np.matmul(C, PC) + S
        K = np.matmul(PC, np.linalg.inv(R))      # PRML 公式 13.92
        Amu = np.matmul(A, mu)
        CAmu = np.matmul(C, Amu)
        mu = Amu + np.matmul(K, x - CAmu)        # PRML 公式 13.89
        V = np.matmul((I - np.matmul(K, C)), P)  # PRML 公式 13.90
        llh = self._logGauss(x, CAmu, R)         # PEML 公式 13.91
        c = np.matmul(C, mu)
        return mu, V, c, llh

    def kalman_smoother(self, X, model):
        '''
          X: 输入观测序列 shape=(features, seq_length) = (128, 60)
        '''
        A = model['A']
        G = model['G']
        C = model['C']
        S = model['S']
        mu0 = model['mu0']
        P0 = model['P0']
        n = X.shape[1]          # 获取序列的长度 (60)
        q = mu0.shape[0]        # 获取隐变量的维数
        mu = np.zeros((q, n))   # 均值
        V = np.zeros((q, q, n)) # 方差（协方差）
        P = np.zeros((q, q, n))
        Amu = np.zeros((q, n))
        llh = np.zeros((1, n))  # 对数似然函数
        I = np.eye(q)
        # forward
        PC = np.matmul(P0, C.T)
        R = np.matmul(C, PC) + S
        K = np.matmul(PC, np.linalg.inv(R))                            # PRML 公式 13.97
        mu[:, 0:1] = mu0 + np.matmul(K, X[:, 0:1] - np.matmul(C, mu0)) # PRML 公式 13.94
        V[:, :, 0] = np.matmul((I-np.matmul(K, C)), P0)                # PRML 公式 13.95
        P[:, :, 0] = P0
        Amu[:, 0:1] = mu0
        llh[0, 0] = self._logGauss(X[:, 0:1], np.matmul(C, mu0), R)    # PRML 公式 13.96
        for i in range(1, n):
            mu[:, i:i+1], V[:, :, i], Amu[:, i:i+1], P[:, :, i], llh[0, i] = self._forward(X[:, i:i+1], mu[:, i-1:i], 
                                                                                        V[:, :, i-1], A, G, C, S, I)
        llh = llh.sum()
        # backward
        nu = np.zeros((q, n))
        U = np.zeros((q, q, n))
        Ezz = np.zeros((q, q, n))   # E[z_tz_t^T]
        Ezy = np.zeros((q, q, n-1)) # E(z_tz_{t-1}^T)
        nu[:, n-1:n] = mu[:, n-1:n]
        U[:, :, n-1] = V[:, :, n-1]
        Ezz[:, :, n-1] = U[:, :, n-1] + np.matmul(nu[:, n-1:n], nu[:, n-1:n].T) # PRML 公式 13.107
        for i in range(n-2, -1, -1):
            nu[:, i:i+1], U[:, :, i], Ezz[:, :, i], Ezy[:, :, i] = self._backward(nu[:, i+1].reshape(-1, 1), 
                                                                                  U[:, :, i+1], 
                                                                                  mu[:, i:i+1], V[:, :, i], 
                                                                                  Amu[:, i+1].reshape(-1, 1), 
                                                                                  P[:, :, i+1], A)
        return nu, U, Ezz, Ezy, llh

    def _forward(self, x, mu0, V0, A, G, C, S, I):
        P = np.matmul(np.matmul(A, V0), A.T) + G                    # PRML 公式 13.88
        PC = np.matmul(P, C.T)
        R = np.matmul(C, PC) + S
        K = np.matmul(PC, np.linalg.inv(R))                         # PRML 公式 13.92
        Amu = np.matmul(A, mu0)
        CAmu = np.matmul(C, Amu)
        mu = Amu + np.matmul(K, x - CAmu)                           # PRML 公式 13.89
        V = np.matmul((I - np.matmul(K, C)), P)                     # PRML 公式 13.90
        llh = self._logGauss(x, CAmu, R)                            # PEML 公式 13.91
        return mu, V, Amu, P, llh

    def _backward(self, nu0, U0, mu, V, Amu, P, A):
        J = np.matmul(np.matmul(V, A.T), np.linalg.inv(P))          # PRML 公式 13.102
        nu = mu + np.matmul(J, nu0 - Amu)                           # PRML 公式 13.100
        U = V + np.matmul(np.matmul(J, U0 - P), J.T)                # PRML 公式 13.101
        Ezz = U + np.matmul(nu, nu.T)                               # PRML 公式 13.107
        Ezy = np.matmul(J, U0) + np.matmul(nu0, nu.T)               # PRML 公式 13.106
        return nu, U, Ezz, Ezy

    def fit(self, X, maxIter=200, tol=1e-6):
        '''
          利用 EM 算法训练模型，得到模型参数（可以适用于多个序列）
          X: 输入观测序列 shape=(samples, features, seq_length)
        '''
        # 对于单个序列情况
        if len(X.shape) == 2:
            model, llh = self.ldsEM(X)
        # 对于有多个序列情况
        elif len(X.shape) == 3:
            model = {}
            model['A'] = self._A
            model['G'] = self._G
            model['C'] = self._C
            model['S'] = self._S
            model['mu0'] = self._mu0
            model['P0'] = self._P0
            # 获取 shape=(1, maxIter) 数组，其中每个元素都是 -inf
            llh = np.array(([-np.inf]*maxIter)).reshape(1, -1)
            iter_num = None
            nums = X.shape[0] # 获取样本个数
            pre_model = model.copy()
            for it in range(1, maxIter+1):
                # E step
                print(it)
                llh_value = 0
                A = np.zeros(self._A.shape)
                G = np.zeros(self._G.shape)
                C = np.zeros(self._C.shape)
                S = np.zeros(self._S.shape)
                mu0 = np.zeros(self._mu0.shape)
                P0 = np.zeros(self._P0.shape)
                for num in range(nums):
                    sample = X[num]
                    # E step
                    nu, U, Ezz, Ezy, llh_temp = self.kalman_smoother(sample, model)
                    # M step
                    result = self._maximization(sample, nu, U, Ezz, Ezy)
                    A += result['A']
                    G += result['G']
                    C += result['C']
                    S += result['S']
                    mu0 += result['mu0']
                    P0 += result['P0']
                    llh_value += llh_temp
                llh[0, it] = llh_value / nums
                model['A'] = A / X.shape[0]
                model['G'] = G / X.shape[0]
                model['C'] = C / X.shape[0]
                model['S'] = S / X.shape[0]
                model['mu0'] = mu0 / X.shape[0]
                model['P0'] = P0 / X.shape[0]
                # 检查似然函数是否收敛 (是否满足停机准则)
                if(llh[0, it] - llh[0, it-1] < tol * np.abs(llh[0, it-1])):
                    iter_num = it
                    model = pre_model.copy()
                    break;
                else:
                    pre_model = model.copy()
            llh = llh[0, 1:iter_num+1]
        return model, llh
    
    def ldsEM(self, X, maxIter=200, tol=1e-6):
        '''
          EM 算法，用于训练模型，得到模型参数(针对单个序列)
          X: 输入观测序列 shape=(features, seq_length) = (128, 60)
        '''
        model = {}
        model['A'] = self._A
        model['G'] = self._G
        model['C'] = self._C
        model['S'] = self._S
        model['mu0'] = self._mu0
        model['P0'] = self._P0
        # 获取 shape=(1, maxIter) 数组，其中每个元素都是 -inf
        llh = np.array(([-np.inf]*maxIter)).reshape(1, -1)
        iter_num = None
        for it in range(1, maxIter+1):
            # E step
            print(it)
            nu, U, Ezz, Ezy, llh[0, it] = self.kalman_smoother(X, model)
            # 检查似然函数是否收敛 (是否满足停机准则)
            if(llh[0, it] - llh[0, it-1] < tol * np.abs(llh[0, it-1])):
                iter_num = it
                break;
            # M step
            model = self._maximization(X, nu, U, Ezz, Ezy)
        llh = llh[0, 1:iter_num+1]
        return model, llh

    def _maximization(self, X, nu, U, Ezz, Ezy):
        n = X.shape[1]           # 获取序列的长度 (60)
        mu0 = nu[:, 0:1]         # PRML 公式 13.110
        P0 = U[:, :, 0]          # PRML 公式 13.111
        Ezzn = Ezz.sum(axis=2)   # 沿着第三个维度求和
        Ezz1 = Ezzn - Ezz[:, :, n-1]
        Ezz2 = Ezzn - Ezz[:, :, 0]
        Ezy = Ezy.sum(axis=2)
        A = np.matmul(Ezy, np.linalg.inv(Ezz1)) # PRML 公式 13.113
        EzyA = np.matmul(Ezy, A.T)
        G = (1./(n-1)) * (Ezz2 - (EzyA + EzyA.T) + np.matmul(np.matmul(A, Ezz1), A.T))          # PRML 公式 13.114
        Xnu = np.matmul(X, nu.T)
        C = np.matmul(Xnu, np.linalg.inv(Ezzn)) # PRML 公式 13.115
        XnuC = np.matmul(Xnu, C.T)
        S = (1./n) * (np.matmul(X, X.T) - (XnuC + XnuC.T) + np.matmul(np.matmul(C, Ezzn), C.T)) # PRML 13.116

        model = {}
        model['A'] = A
        model['G'] = G
        model['C'] = C
        model['S'] = S
        model['mu0'] = mu0
        model['P0'] = P0
        return model

def _gaussRnd(mu, sigma):
    '''
    产生样本数据(一次产生一个)，满足高斯分布
    用于测试目的
    '''
    V = np.linalg.cholesky(sigma).T  # 获取 sigma 的上三角矩阵
    x = np.matmul(V.T, np.random.randn(V.shape[0], 1)) + mu
    return x

def get_samples_datas(classify_object_name=0):
    if classify_object_name == 0:
        path = "./samples_features/valence_without_features_smooth/"
    elif classify_object_name == 1:
        path = "./samples_features/arousal_without_features_smooth/"
    samples_dirs = os.listdir(path)
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    people_list = list(range(32))
    all_datas_list = []
    for people in people_list:
        datas = np.load(file_path[people]+"/datas.npy")
        assert(len(datas)==40)
        assert(datas[0].shape==(128, 60))
        all_datas_list.append(np.array(datas))
    result = np.r_[tuple(all_datas_list)]
    assert(result.shape==(1280, 128, 60))
    return result

def moving_average(X):
    assert(X.shape==(128, 60))
    temp = X.copy()
    nums = 60 - 20 + 1
    for num in range(nums):
        X[:, num+20-1] = temp[:, num:(num+20)].mean(axis=1)
    for i in range(20-1):
        X[:, i] = temp[:, 0:i+1].mean(axis=1)
    assert(X.shape==(128, 60))
    return X

if __name__ == "__main__":
    datas = get_samples_datas(classify_object_name=0)
    mv_data = moving_average(datas[8].copy())
    # lds = LDS(d=128, k=2, model=None)
    # model, llh = lds.fit(datas, tol=1e-3)
    # _, data = lds.kalman_filter(datas[8], model)
    # print(model)
    # print(llh)
    # np.save("./lds_model_params/model_params_43", model)
    model = np.load("./lds_model_params/model_params_31.npy").item()
    data = linear_dynamical_systems(datas, 60, model)
    plt.plot(datas[8][0], label="raw")
    plt.plot(data[8][0], label="lds")
    plt.plot(mv_data[0], label="mv")
    plt.legend()
    plt.grid()
    plt.show()
