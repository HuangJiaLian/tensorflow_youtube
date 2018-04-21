
# coding: utf-8

# In[21]:


import tensorflow as tf

x = tf.Variable([1,2])
a = tf.constant([3,3])

# add a subtract operation
sub = tf.subtract(x,a)

# add a add operation 
add = tf.add(x,sub)

init  = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


# In[22]:



# create a variable 
state = tf.Variable(0, name='counter')
# crate a operation to add 
new_value = tf.add(state, 1)
# assige new_value to state
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))

