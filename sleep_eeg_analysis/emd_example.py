import numpy as np
import matplotlib.pyplot as plt
import PyEMD

np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time) + 1
s2 = np.sin(5 * time)
S = s1 + s2
S = (S - S.min()) / (S.max() - S.min())

ceemd = PyEMD.EMD()
#IMF = ceemd.ceemdan(S)
IMF = ceemd.emd(S)
print("IMF shape: ", IMF.shape)
N = IMF.shape[0] + 1

S1 = IMF.T.sum(axis=1)
print("S1 shape: ", S1.shape)

plt.figure(1)
#plt.subplot(N, 1, 1)
plt.plot(S)
plt.grid()
plt.figure(3)
for n, imf in enumerate(IMF):
    plt.subplot(N, 1, n+2)
    plt.plot(imf)
    plt.grid()
plt.xlabel("Time [s]")
plt.figure(2)
plt.plot(S1)
plt.grid()
plt.show()
