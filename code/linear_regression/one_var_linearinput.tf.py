#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_FILE_NAME = './dataset/1_linearinput.csv'

# load data
df = pd.read_csv(DATA_FILE_NAME)
# plot data
df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# expected line
plt.plot(df.values[:,0], df.values[:,1], color='g')
plt.xlabel('x'); plt.ylabel('y'); plt.show()

# extract X, Y
x_train = df.values[:,0].tolist()
y_train = df.values[:,2].tolist()

# Model parameters
W = tf.Variable([0], dtype=tf.float32)
b = tf.Variable([0], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(2):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# plot result
# training data
df.plot(x='x', y='y', legend=False, marker='o', style='o', mec='b', mfc='w')
# expected line
plt.plot(df.values[:,0], df.values[:,1], color='g')
# predicted line
X_hat = np.linspace(df.values[:,0].min(), df.values[:,0].max(), 10)
Y_hat = curr_b + curr_W * X_hat
plt.plot(X_hat, Y_hat, color='r')
plt.xlabel('x'); plt.ylabel('y'); plt.show()
