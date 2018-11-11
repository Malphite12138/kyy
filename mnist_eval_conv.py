import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_conv
import mnist_train_conv
import numpy as np

#每10秒加载一次最新的模型
EVAL_INTERVAL_SECS = 300

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式
        x = tf.placeholder(tf.float32, [mnist_train_conv.BATCH_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.IMAGE_SIZE,
                                   mnist_inference_conv.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None,mnist_inference_conv.OUTPUT_NODE], name = 'y_input')
        #validate_feed = {x:mnist.validation.images,
         #               y_:mnist.validation.labels}
        #xs = mnist.validation.images
        #ys = mnist.validation.labels
        #reshaped_xs = np.reshape(validate_feed, (mnist_train_conv.BATCH_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.IMAGE_SIZE,
         #                             mnist_inference_conv.NUM_CHANNELS))
        #validate_feed = {x: reshaped_xs,y_: ys}
        #测试的时候不考虑正则化损失，所以正则化为空

        xs, ys = mnist.validation.next_batch(mnist_train_conv.BATCH_SIZE)
        reshaped_xs = np.reshape(xs, (mnist_train_conv.BATCH_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.IMAGE_SIZE,
                                       mnist_inference_conv.NUM_CHANNELS))
        validate_feed = {x:reshaped_xs,y_:ys}

        y = mnist_inference_conv.inference(x,True,None)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


        variable_averages = tf.train.ExponentialMovingAverage(mnist_train_conv.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state('ckpt/')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)

                    print("第 %s 次训练后，测试精度是 %g" % (global_step,accuracy_score))
                else:
                    print("找不到文件")
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets("D:\MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()