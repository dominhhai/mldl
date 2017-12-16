#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_NAME = './dataset/1_sin2pi.csv'
DEGREE = 3 # degree of polynomial of X
ETA = .01 # learning rate
EPSILON = 1e-6 # gradient stop minimum condition
LAMBDA = 1 # regulization hyperparameter
MAX_STEP = 1000000 # gradient loop step
GRAD_CHECK_EPSILON = 1e-10
GRAD_CHECK_MAX = 1e-4

"""
calculate gradient of cost function
```
grad = sum((wx - y)x)
```
"""
def calcGrad(W, X, Y):
    grad = np.zeros_like(W)
    m = X.shape[0]
    for i in range(m):
        x = X[i,:].T
        grad += np.dot(np.dot(W.T, x) - Y[i], x)
    return grad

"""
calculate numerial gradient
```
numerialGrad = (f(w+h) - f(w-h)) / 2h
```
"""
def calcNumericalGrad(W, X, Y):
    grad = np.zeros_like(W)
    m = X.shape[0]
    for i in range(grad.shape[0]):
        H = np.zeros_like(W)
        H[i] = GRAD_CHECK_EPSILON
        for j in range(m):
            grad[i] += (np.square(np.dot(X[j,:],W+H)-Y[j]) - np.square(np.dot(X[j,:],W-H)-Y[j])) / (4 * GRAD_CHECK_EPSILON)

    return grad

# load data
df = pd.read_csv(DATA_FILE_NAME)
# plot data
df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# # expected line
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
W = np.zeros(_X.shape[1])

for i in range(MAX_STEP):
    grad = calcGrad(W, _X, Y)
    # check gradient is correct
    if i == 0:
        numerialGrad = calcNumericalGrad(W, _X, Y)
        if np.linalg.norm(numerialGrad - grad) > GRAD_CHECK_MAX:
            print numerialGrad, grad, np.linalg.norm(numerialGrad - grad)
            print 'Gradient Calc Error!'
            break
    # regulization
    w = LAMBDA * (np.sum(W) - W[0])
    grad += w
    grad[0] -= w
    # update parameters
    W -= ETA * grad
    # Exit when gradient is small enought
    gradLen = np.linalg.norm(grad)
    print i, W, gradLen
    if gradLen <= EPSILON:
        print gradLen
        break

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
