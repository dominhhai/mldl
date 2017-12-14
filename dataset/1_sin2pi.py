#!/usr/bin/python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIST_FILE_NAME = os.path.splitext(__file__)[0] + '.csv'
DATA_SIZE = 100
N_SIGMA = 0.15

# data[X, standard Y, noise Y]
data = np.empty((DATA_SIZE, 3))
# input vector X
data[:,0] = np.random.rand(DATA_SIZE)
data[:,0].sort()
# output vector Y=sin(2pi*X)
data[:,1] = np.sin(2 * np.pi * data[:,0])
# make noise by normal distribution: mu = Y
data[:,2] = N_SIGMA * np.random.randn(DATA_SIZE) + data[:,1]

# build DataFrame
df = pd.DataFrame(data=data, columns=list('xsy'))
# write down to CSV file
df.to_csv(DIST_FILE_NAME, index=False)
print('... y=sin(2pi * x) data ...')

# plot
df.plot(x='x', title='y=sin(2pi * x)')
# df.plot(x='x', y='y', legend=False, marker='o', color='w', mec='b', mfc='w')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
