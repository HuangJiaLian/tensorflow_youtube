
# coding: utf-8

# In[1]:


# ** Running Process Visualise **

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LAYER1_NODE = 500

# Load data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


batch_size = 20
# Number of batch 
n_batch = mnist.train.num_examples // batch_size

# ** Running Process Visualise **
# Calculate something
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

# ** Visualize **
# Name Space to Visualise
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None, 784], name='x-input')
    y = tf.placeholder(tf.float32,[None, 10], name='y-input') # Lable

# ** Visualize **
with tf.name_scope('layer'):
    # Build Network
    with tf.name_scope('wight'):
        W = tf.Variable(tf.zeros([784,10]), name = 'W')
        # ** Running Process Visualise **
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name = 'b')
        # ** Running Process Visualise **
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)


# quadratic cost
# loss = tf.reduce_mean(tf.square(y-prediction))

# ** Visualize **
with tf.name_scope('loss'):
    # 交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # ** Running Process Visualise **
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()


# ** Visualize **
with tf.name_scope('accuracy'): 
    with tf.name_scope('correct_prediction'):
        # correct_prediction：A boolean list
        # argmax: return the index of max value 
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        # Get accuracy rate: 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # ** Running Process Visualise **
        tf.summary.scalar('accuracy',accuracy)

# ** Running Process Visualise **
# Combine all summarise
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # ** Visualize **
    # tensorboard --logdir= ./logs/
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            # Store image in batch_xs, lable in batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # ** Running Process Visualise **
            # Add a fetch opration
            summary, _ = sess.run([merged,train_step], feed_dict={x:batch_xs, y:batch_ys})
        # ** Running Process Visualise **
        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(acc))

