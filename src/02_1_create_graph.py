
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


# 创建两个常量op
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

# 创建一个矩阵乘法的op
product = tf.matmul(m1,m2)

print(product)


# In[3]:


# 定义一个会话，启动默认图
sess = tf.Session()
# 调用sess的run方法
result = sess.run(product)
print(result)
sess.close()


# In[5]:


# 这样就不用关闭会话了，运行结束会自动关闭．
with tf.Session() as sess:
    # 定义一个会话，启动默认图
    sess = tf.Session()
    # 调用sess的run方法
    result = sess.run(product)
    print(result)

