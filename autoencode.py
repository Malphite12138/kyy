import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:\MNIST_data", one_hot=False)

LEARNING_RATE = 0.01
TRAINING_EPOCHS = 20
BATCH_SIZE = 256
DISPLAY_STEP = 10
N_INPUT = 784
IMAGE_SHOW = 10

x = tf.placeholder("float", [None,N_INPUT])

N_HIDDEN1 = 256
N_HIDDEN2 = 128

ENCODER_H1 = tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN1]))
ENCODER_H2 = tf.Variable(tf.random_normal([N_HIDDEN1, N_HIDDEN2]))
DECODER_H1 = tf.Variable(tf.random_normal([N_HIDDEN2, N_HIDDEN1]))
DECODER_H2 = tf.Variable(tf.random_normal([N_HIDDEN1, N_INPUT]))
"""
WEIGHTS = {
    'ENCODER_H1'
}
"""
ENCODER_B1 = tf.Variable(tf.random_normal([N_HIDDEN1]))
ENCODER_B2 = tf.Variable(tf.random_normal([N_HIDDEN2]))
DECODER_B1 = tf.Variable(tf.random_normal([N_HIDDEN1]))
DECODER_B2 = tf.Variable(tf.random_normal([N_INPUT]))

def encoder(x):
    layer1 = tf.nn.sigmoid(tf.matmul(x, ENCODER_H1) + ENCODER_B1)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, ENCODER_H2) + ENCODER_B2)
    return layer2

def decoder(x):
    layer1 = tf.nn.sigmoid(tf.matmul(x, DECODER_H1) + DECODER_B1)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, DECODER_H2) + DECODER_B2)
    return layer2

encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = x

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))   #最小二乘法求损失

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:

    #if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).spilt('.')[0]) < 1:
    init = tf.initialize_all_variables()
    #else:
    #    init = tf.global_variables_initializer()
    sess.run(init)

    total_batch = int(mnist.train.num_examples / BATCH_SIZE)
    #print(total_batch)
    for epoch in range(TRAINING_EPOCHS):
        print(epoch)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs})

        if epoch % DISPLAY_STEP == 0:
            print("到第 %d 次后，损失为 %s"  %(epoch+1, "{:.9f}".format(c)))
    print("优化完成")


    encode_decode = sess.run(y_pred, feed_dict={x: mnist.test.images[:IMAGE_SHOW]})
    f, a = plt.subplots(2, 10, figsize=(10,2))
    for i in range(IMAGE_SHOW):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
    plt.show()
















