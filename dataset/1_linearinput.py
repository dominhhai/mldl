#!/usr/bin/python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIST_FILE_NAME = os.path.splitext(__file__)[0] + '.csv'
DATA_SIZE = 100
N_SIGMA = 1500

# data[X, standard Y, noise Y]
data = np.empty((DATA_SIZE, 3))
# input vector X
data[:,0] = np.random.random_integers(1000, 5000, DATA_SIZE)
data[:,0].sort()
# output vector Y=4X+3
data[:,1] = 4 * data[:,0] + 3
# make noise by normal distribution: mu = Y
data[:,2] = N_SIGMA * np.random.randn(DATA_SIZE) + data[:,1]

# build DataFrame
df = pd.DataFrame(data=data, columns=list('xsy'))
# write down to CSV file
df.to_csv(DIST_FILE_NAME, index=False)
print('... y=4x+3 data ...')

# plot
df.plot(x='x', title='y=4x+3')
# df.plot(x='x', y='y', legend=False, marker='o', color='w', mec='b', mfc='w')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
