#!/usr/bin/python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIST_FILE_NAME = os.path.splitext(__file__)[0] + '.csv'
N_SIGMA = 550

# input vector x
X = np.random.random_integers(1000, 2000, 100)
X.sort()
# output vector Y=4X+3
Y = 4*X + 3
# make noise by normal distribution: mu = y
Y = N_SIGMA * np.random.randn(Y.shape[0]) + Y

# build DataFrame
df = pd.DataFrame(data=Y, index=X, columns=['y'])
# write down to CSV file
df.to_csv(DIST_FILE_NAME, header=False)
print('... 4x+3 data ...')

# plot
df.plot(legend=False, marker='o', color='w', mec='b', mfc='w')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
