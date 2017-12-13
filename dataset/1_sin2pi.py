#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIST_FILE_NAME = '1_sin2pi.csv'

# input vector x
x = np.random.rand(50)
x.sort()
# output vector y=2pi * x
y = np.sin(2 * np.pi * x)

# build DataFrame
df = pd.DataFrame(data=y, index=x, columns=['y'])
# write down to CSV file
df.to_csv(DIST_FILE_NAME, header=False)
print('... sin(2*pi) data ...')

# plot
df.plot(legend=False, marker='o', color='g', mec='b', mfc='w')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
