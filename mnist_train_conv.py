import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#import mnist_inference_conv
import mnist_inference_conv

#配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99

#保存模型的路径和文件名
#MODEL_NAME = "model.ckpt"
#MODEL_PATH = "/path/to/model/"

def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.IMAGE_SIZE,
                                   mnist_inference_conv.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference_conv.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    #使用inference中定义的前向传播过程
    y = mnist_inference_conv.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.IMAGE_SIZE,
                                         mnist_inference_conv.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("第 %d 次训练后，损失是 %g" % (step-1, loss_value))
                #saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME),global_step = global_step)
                saver.save(sess, 'ckpt/model.ckpt',global_step = global_step)

def main(argv = None):
    mnist = input_data.read_data_sets("D:\MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
