import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))

S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)

S /= S.std(axis=0)
A = np.array([[1,1], [0.5, 2]])
X = np.dot(S, A.T)

ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)
A_ = ica.mixing_
S_remove = S_.copy()
S_remove[:,1] = np.zeros(S_remove[:,1].shape)
print(S_remove)
print(S_)
X_ = ica.inverse_transform(S_remove)
plt.figure(3)
plt.plot(X)
plt.figure(2)
plt.plot(X_)
plt.figure(1)
models = [X, S, S_]
names = ['observations(mixed signal)', 'True Sources', 'ICA recovered signals']
colors = ['red', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.show()
print(X.shape)
print(S.shape)
print(S_.shape)
