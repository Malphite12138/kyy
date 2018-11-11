import tensorflow as tf

#配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和参数
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二层
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor,train,regularizer):

    #第一层卷积层，输入是28*28*1，输出是28*28*32.其余代码解释，见课本 P147
    with tf.variable_scope('layer1-conv1'):

        #创建过滤器的权重变量，第一个参数是名称；第二个参数是一个矩阵，前两个维度指的是卷积层过滤器的尺寸，
        #第三个维度是指当前层的深度（暂时认为是图片的第三个维度，黑白为1，彩色为3。因为这是第一层卷积层，后边就不一样了），
        #第四个维度是指卷积层过滤器的深度
        conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))

        #卷积层的偏置项，第二个参数是下一层节点矩阵的深度（暂时认为是卷积层过滤器的深度）
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        #第一个参数是当前层的节点矩阵，这个矩阵是四维的，第一维是对应一个输入的batch，后三个维度对应一个节点矩阵[0,:,:,:]表示第一张图片
        #第二个参数是卷积层的权重；第三个参数是不同维度上的步长，但是要求第一维和最后一维都是1
        #第四个参数表示是否填充0，SAME为填充，VALID表示不填充
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')

        #bias_add()对每个节点加上一个偏置，relu对它进行去线性化
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #第二层池化层，输入28*28*32，步长为2，所以输出14*14*32
    with tf.name_scope('layer2-pool1'):

        #tf.nn.max_pool()的第一个参数就是当前层的节点矩阵，也就是上一层的输出。第二个参数是过滤器的尺寸，第一维和最后一
        #维必须是1。第三个参数是步长，同样一头一尾都是1.第四个参数指定是否填充0
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #第三层卷积层，输入为14*14*32，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #第四层池化层，输入为14*14*64，输出为7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #第五层为全连接层，全连接层的输入是一个向量，但是第四层池化层输出的是一个矩阵，所以使用函数进行拉伸
    #每一层网络的输入输出都是一个batch的矩阵，所以通过函数拉伸得到的维度也包括了一个batch中数据的个数
    pool_shape = pool2.get_shape().as_list()

    #将矩阵拉伸成向量之后的大小就是矩阵长宽以及深度的乘积
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    #通过reshape把第四层的输出变成一个batch的向量
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    #第五层全连接层，全连接的输入是一组向量，输出也是一组向量，在本代码中，输出向量的长度为512
    #这一层中引入了dropout算法，在训练时随机将部分节点的输出改为0。
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    #第六层全连接层，输入为上一层的输出，一组长度为512的向量，输出长度为10，这层的输出通过softmax之后就是最终结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight",[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights) + fc2_biases
    return logit