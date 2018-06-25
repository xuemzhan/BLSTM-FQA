import tensorflow as tf
import numpy as np


def softmax(input_tensor):
    """
    用来处理变长的sorfmax函数，常用于attention层的变换
    :param input_tensor:要处理的tensor，默认为要处理最后一维
    :return: 和input_tensor的shape 一样的tensor
    """
    mask = tf.sign(tf.abs(input_tensor))
    exp_tensor = tf.exp(input_tensor)
    exp_tensor = tf.multiply(exp_tensor, mask)
    exp_sum = tf.reduce_sum(exp_tensor, reduction_indices=1)
    exp_tensor = tf.transpose(exp_tensor)
    softmax_output = tf.divide(exp_tensor, exp_sum)
    softmax_output = tf.transpose(softmax_output)
    return softmax_output

if __name__ == '__main__':
    data = np.array([[1, 2, 0], [2, 3, 4], [3, 0, 0]])
    data = tf.constant(data, dtype=tf.float32)
    softmax_out = softmax(data)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([softmax_out]))
