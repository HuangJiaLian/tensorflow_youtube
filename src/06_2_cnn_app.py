# coding: utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import cv2
import os

PaintSize = 350
BgColor = 255
PaintColor = 0
StrokeWeight = 20

drawing = False
start = (-1, -1)
lastPoint = (-1,-1)

# Initialise background color to white.
img = np.full((PaintSize, PaintSize,1), BgColor, dtype=np.uint8)

# mouse callback
def mouse_event(event, x, y, flags, param):
    global drawing, img, lastPoint
    # Left button down
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        lastPoint = (x, y)
        start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img,lastPoint,(x, y), PaintColor, StrokeWeight)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    lastPoint = (x, y)


# Image Processing Step
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28,28), True) # Reshape the image to the size of 28x28

    im_arr = np.array(reIm.convert('L')) # convert to grayscale 
    cv2.imshow("Little",im_arr)
    threshold = 50

    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            # if (im_arr[i][j] < threshold):
            #     im_arr[i][j] = 0
            # else:
            #     im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    print(img_ready)
    return img_ready



# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variablle(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x,W):
    # x是一个4D的tensor: [batch, in_height, in_width, in_channels]
    # W是卷积核的属性: [filter_height, filter_width, in_channels, out_channels]
    # strides: 必须要第一个和最后一个相同 `strides[0] = strides[3] = 1`. 
    # 大多数情况下，水平和垂直方向的strides取一样的值，结构就像下面这样
    # `strides = [1, stride, stride, 1]`.
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# 池化层
def max_pool_2x2(x):
    # 核的形状：[ 1,x,y,1] 第一个 和 最后一个元素为1， x,y为核的大小
    # strides: 必须要第一个和最后一个相同 `strides[0] = strides[3] = 1`. 
    # 大多数情况下，水平和垂直方向的strides取一样的值，结构就像下面这样
    # `strides = [1, stride, stride, 1]`.
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def restore_model(testPicArr):
    MODEL_SAVE_PATH = './model/'
    MODEL_NAME='cnn_mnist_model' 
    fc_node = 500
    with tf.Graph().as_default() as tg:
        # 定义两个placeholder
        x = tf.placeholder(tf.float32,[None, 784]) # 第一个数字代表行，784代表有784列
        # y = tf.placeholder(tf.float32,[None, 10])  # 输出：标签

        # 改变x为4D: [batch, in_height, in_width, in_channels]
        x_image = tf.reshape(x, [-1,28,28,1]) # -1：现在不关心，后续会变成100

        # 初始化第一个卷积层
        # 卷积核的形态：5*5*1
        # 卷积核的个数：32 (这个数字是可以尝试出来的是吧？)
        # 用32个卷积核去对一个平面/通道采样，最后会得到32个卷积特征平面
        W_conv1 = weight_variable([5,5,1,32]) 
        b_conv1 = bias_variablle([32]) 

        # 把x_image和卷积向量进行卷积，再加上偏置，然后应用于relu激活函数
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # 初始化第二个卷积层
        # 卷积核的形态：5*5*32 
        # 卷积核的个数：64
        # 使用64个卷积核对32个平面提取特征；得到64x32个特征平面 (他说是64) ??
        # 回答：的确是64个，卷的时候是考虑了深度的，卷积核在这里考虑成一个cube(立方体)
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variablle([64]) # 一个卷积核要一个偏置值

        # 把h_pool1和卷积向量进行卷积，再加上偏置，然后应用于relu激活函数
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # 28x28的图片第一次卷积后还是28x28(same padding)，第一次池化后变14x14(池化窗口2x2)
        # 第二次卷积后为14x14,第二次池化后变为7x7
        # 通过上面的操作得到64张7x7的平面

        # 初始化第一个全连接层的权值
        W_fc1 = weight_variable([7*7*64,fc_node]) # 上一层有7x7x64个神经元，定义全连接层有1024个神经元
        b_fc1 = bias_variablle([fc_node]) # 1024个节点

        # 把池化层的输出扁平化为1维
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) # 4D->2D
        # 求第一个全连接层的输出
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # keep_prob用来表示神经元的输出概率
        # 也就是一次训练中只使用百分之多少的神经元
        # 用来避免过拟合
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        # 初始化第二个全连接层
        W_fc2 = weight_variable([fc_node,10])
        b_fc2 = bias_variablle([10])

        # 计算输出
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
        preValue = tf.argmax(prediction,1)

        # saver 用来保存训练模型
        saver = tf.train.Saver()

        with tf.Session() as sess:   
                model_path = os.path.join(MODEL_SAVE_PATH,MODEL_NAME)
                saver.restore(sess, model_path)
                preValue = sess.run(preValue, feed_dict={x:testPicArr,keep_prob:1.0})
                return preValue

def application():
    global img
    cv2.namedWindow('Press \'s\' to Save,\'c\' to clear')
    cv2.setMouseCallback('Press \'s\' to Save,\'c\' to clear', mouse_event)
    print("Press q or Esc to quit the program:")
    while True:
        cv2.imshow('Press \'s\' to Save,\'c\' to clear', img)
        key = cv2.waitKey(20)
        if key == 27 or key == 113: # break when press ESC/q
            break
        elif key == 115: # s for 'save'
            imgName = './pic/handWrite.png'
            cv2.imwrite(imgName, img)
            print(imgName + " saved")
            testPicArr = pre_pic(imgName)
            preValue = restore_model(testPicArr)
            print ("The prediction number is:", preValue)
        elif key == 99: # c for 'clear'
            img = np.full((PaintSize, PaintSize,1), BgColor, dtype=np.uint8)
        else:
            pass

def main():
    application()
    
if __name__ == '__main__':
    main()
