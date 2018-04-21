
# coding: utf-8

# In[6]:


import tensorflow as tf
import numpy as np
# Generate 100 samples
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# Build a linear model
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b 

# Loss function 
loss = tf.reduce_mean(tf.square(y_data - y))

# Define the optimizer to train model
# 0.2: Learning rate
optimizer = tf.train.GradientDescentOptimizer(0.2)

# The object of our algorithm
train = optimizer.minimize(loss)


# Build a new operation intend to initialize all variables 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run([k,b]))
            

