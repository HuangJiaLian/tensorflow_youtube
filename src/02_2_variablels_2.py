
# coding: utf-8

# In[19]:


import tensorflow as tf

# create a variable 
state = tf.Variable(0, name='counter')
# crate a operation to add 
new_value = tf.add(state, 1)
# assige new_value to state
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()

with tf.Session as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))

