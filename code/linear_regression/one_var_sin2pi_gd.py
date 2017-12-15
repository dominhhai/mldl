#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_NAME = './dataset/1_sin2pi.csv'
DEGREE = 3 # degree of polynomial of X
ETA = 1e-3 # learning rate
EPSILON = 1e-6 #0.000006 gradient stop minimum condition
MAX_STEP = 100000 # gradient loop step
GRAD_CHECK_EPSILON = 1e-10
GRAD_CHECK_MAX = 1e-6

# load data
df = pd.read_csv(DATA_FILE_NAME)
# plot data
# df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# # expected line
# plt.plot(df.values[:,0], df.values[:,1], color='g')
# plt.xlabel('x'); plt.ylabel('y'); plt.show()

# extract X, Y
X = df.values[:,0]
Y = df.values[:,2]
# build polynomial X
_X = np.empty((X.shape[0], DEGREE+1))
for i in range(DEGREE + 1):
    _X[:,i] = np.power(X,i)
# print _X.shape, Y.shape

# parameters learning
W = np.zeros(_X.shape[1])
grad = np.zeros(W.shape[0])
# print W.shape, _X.shape, Y.shape
for i in range(MAX_STEP):
    for j in range(X.shape[0]):
        grad += np.dot(np.dot(W.T,_X[j,:])-Y[j], _X[j,:].T)
    gradLen = np.linalg.norm(grad)

    # if i == 0:
    #     _grad = np.empty(W.shape[0])
    #     for j in range(grad.shape[0]-1):
    #         _gradEps = np.zeros(W.shape[0])
    #         _gradEps[j] = GRAD_CHECK_EPSILON
    #         _grad[j] = (np.sum(np.square(np.dot(_X,W+_gradEps)-Y))-np.sum(np.square(np.dot(_X,W-_gradEps)-Y))) / (4 * GRAD_CHECK_EPSILON)
    #     if np.linalg.norm(grad - _grad) > GRAD_CHECK_MAX:
    #         print 'Gradient Calc Error!'
    #         break

    if gradLen <= EPSILON:
        print gradLen
        break
    W -= ETA * grad / X.shape[0]

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
