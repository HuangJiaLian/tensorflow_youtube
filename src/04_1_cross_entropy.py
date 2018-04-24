
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LAYER1_NODE = 2000
LAYER2_NODE = 2000
LAYER3_NODE = 1000
LAYER4_NODE = 1000
# Load data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 200
# Number of batch 
n_batch = mnist.train.num_examples // batch_size

# Define placeholder
x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10]) # Lable
keep_prob = tf.placeholder(tf.float32) # How much percentage of node is working


# Build Network
# # W1 = tf.Variable(tf.zeros([784,10]))
# W1 = tf.Variable(tf.truncated_normal([784,10], stddev=0.1))
# # b1 = tf.Variable(tf.zeros([10]))
# b1 = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(x,W1)+b1)

W1 = tf.Variable(tf.truncated_normal([784,LAYER1_NODE],stddev=0.1))
b1 = tf.Variable(tf.zeros([LAYER1_NODE]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,LAYER2_NODE],stddev=0.1))
b2 = tf.Variable(tf.zeros([LAYER2_NODE]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([LAYER2_NODE,LAYER3_NODE],stddev=0.1))
b3 = tf.Variable(tf.zeros([LAYER3_NODE]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)


W4 = tf.Variable(tf.truncated_normal([LAYER3_NODE,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)


# quadratic cost
# loss = tf.reduce_mean(tf.square(y-prediction))
# cross entropy loss function 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()


# correct_prediction：A boolean list
# argmax: return the index of max value 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
# Get accuracy rate: 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            # Store image in batch_xs, lable in batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels, keep_prob:0.7})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images,y:mnist.train.labels, keep_prob:0.7})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc),", Training Accuracy " + str(train_acc))


# In[ ]:


# keep_prob = 1.0 
# Iter 0, Testing Accuracy 0.9344 , Training Accuracy 0.94698185
# Iter 1, Testing Accuracy 0.9501 , Training Accuracy 0.96681815
# Iter 2, Testing Accuracy 0.9551 , Training Accuracy 0.9756727

