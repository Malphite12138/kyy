import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784    #输入节点的个数，也就是图片的像素，因为用的是手写数字集，所以输入节点为784
OUTPUT_NODE = 10    #输出节点个数，因为手写数字集是十个类别，所以输出节点个数是十

#配置参数
LAYER1_NODE = 500  #隐藏层节点个数
BATCH_SIZE = 100   #一个batch的大小

LEARNING_RATE_BASE = 0.8   #基础学习率
LEARNING_RATE_DECAY = 0.99   #学习率的衰减率

REGULARIZATION_RATE = 0.0001   #描述模型复杂度的正则化项在损失函数中的系数
TRAIN_STEPS = 30000   #训练次数
MOVING_AVERAGE_DECAY = 0.99   #滑动平均衰减率


def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:   #如果没有滑动平均类的时候
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
        return tf.matmul(layer1,weights2) + biases2

    else:   #如果有滑动平均类的时候，就先把权重和偏置进行滑动平均之后再进行计算
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

#模型训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None,INPUT_NODE], name = 'x_input')
    y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name = 'y_input')

    #生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    #输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0,trainable=False)   #定义一个用来储存训练轮数的变量，在使用TensorFlow进行训练时，一般将代表训练轮数的参数设置为不可训练参数

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)


    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)


    #求交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_, 1))

    #计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    #总损失使交叉损失和正则化损失的和
    loss = regularization + cross_entropy_mean

    #学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

#使用优化器优化算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

#反向传播更新参数
    train_op = tf.group(train_step, variables_averages_op)

#判断两个张量的每一维是否相等，相等返回true，否则false
    correct_predicition = tf.equal(tf.argmax(average_y, 1),tf.argmax(y_, 1))

#先将数值转换为实数型，然后计算平均值，这个平均值就是正确率
    accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

#初始化训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

    #条件和评判训练效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    #测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    #训练
        for i in range(TRAIN_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("第 %d 次训练后，精度是 %g" % (i, validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
    print("第 %d 次训练后，测试精度是 %g" % (TRAIN_STEPS,test_acc))

#主程序
def main(argv=None):
    mnist = input_data.read_data_sets("D:\MNIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()