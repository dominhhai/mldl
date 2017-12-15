#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

DATA_FILE_NAME = './dataset/1_linearinput.csv'

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
# print X.shape, Y.shape

# parameters learning
reg = linear_model.LinearRegression()
reg.fit(X, Y)
W_hat = reg.coef_
print 'W_hat :=',W_hat

# plot result
# training data
df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# expected line
plt.plot(df.values[:,0], df.values[:,1], color='g')
# predicted line
X_hat = np.linspace(X[:,1].min(), X[:,1].max(), 10)
Y_hat = W_hat[0] + W_hat[1] * X_hat
plt.plot(X_hat, Y_hat, color='r')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
