#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_NAME = './dataset/1_sin2pi.csv'

# load data
df = pd.read_csv(DATA_FILE_NAME, index_col=0, header=None, names=['y'])

# plot
df.plot(legend=False, marker='o', color='g', mec='b', mfc='w')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
