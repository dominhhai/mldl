#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_NAME = './dataset/1_sin2pi.csv'

# load data
df = pd.read_csv(DATA_FILE_NAME, index_col=0, header=None, names=['y'])
# plot training data
# df.plot(legend=False, marker='o', color='g', mec='b', mfc='w')
# plt.xlabel('x'); plt.ylabel('y'); plt.show()

# parameters learning
x = np.array([df.index.values, np.ones(df.shape[0])])
y = df.values
print(x.shape)
print(y.shape)
w_hat = np.matmul(np.matmul(np.transpose(np.matmul(x.T,x)),x.T),y)
print(w_hat.shape)
y_hat = np.matmul(w_hat.T,x)
print(y_hat.shape)
