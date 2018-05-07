
# coding: utf-8

# In[ ]:


# ** projector **
# Tips: Change you DIR at first
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# Load data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

max_steps = 3001
image_num = 5000
batch_size = 100

DIR = os.getcwd()+'/'
# print(DIR)
# Define Session
sess = tf.Session()

# Load images (From the first image to the image_num th image)
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

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

# Name Space to Visualise
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None, 784], name='x-input')
    y = tf.placeholder(tf.float32,[None, 10], name='y-input') # Lable
    
# Show Images
with tf.name_scope('imput_reshape'):
    # -1: uncertain number 1: Demension(GrayScale)
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input,10) # 10 images

    
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

# ** Visualize **
with tf.name_scope('loss'):
    # 交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # ** Running Process Visualise **
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# Initialize
sess.run(tf.global_variables_initializer())


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

# ** projector **
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.Remove(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv','w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# Combine all summarise
merged = tf.summary.merge_all()

# ** projector **
projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

for i in range(max_steps):
    # Store image in batch_xs, lable in batch_ys
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # ** projector **
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged,train_step], feed_dict={x:batch_xs, y:batch_ys}, options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary,i)
    
    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(i) + ", Testing Accuracy " + str(acc))

saver.save(sess,DIR + 'projector/projector/a_model.ckpt',global_step=max_steps)
projector_writer.close()
sess.close()


