#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_NAME = './dataset/1_sin2pi.csv'
DEGREE = 3 # degree of polynomial of X

# load data
df = pd.read_csv(DATA_FILE_NAME)
# plot data
df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# expected line
plt.plot(df.values[:,0], df.values[:,1], color='g')
plt.xlabel('x'); plt.ylabel('y'); plt.show()

# extract X, Y
X = df.values[:,0]
Y = df.values[:,2]
# build polynomial X
_X = np.empty((X.shape[0], DEGREE+1))
for i in range(DEGREE + 1):
    _X[:,i] = np.power(X,i)
# print _X.shape, Y.shape

# parameters learning
# W = np.dot(np.linalg.pinv(X), Y)
W = np.dot(np.linalg.pinv(np.dot(_X.T,_X)), np.dot(_X.T, Y))
print 'W :=', W

# plot result
# training data
df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# expected line
plt.plot(X, df.values[:,1], color='g')
# predicted line
Y_hat = np.dot(_X,W)
plt.plot(X, Y_hat, color='r')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
