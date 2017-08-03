import tensorflow as tf
import numpy as np
from ...utils.tools import Tools

log = Tools.get_logger('CNNText')

class CNNText:
    def __init__(self, X, y, 
            class_num=1999, 
            kernel_lens=[3, 4, 5, 6], 
            num_outputs=512, 
            embedding_size=256,
            learning_rate=0.01,
            regularizer_scale=0.1):
        self.X = tf.expand_dims(X, -1)
        self.y = y
        self.class_num = class_num
        self.kernel_lens = kernel_lens
        self.embedding_size = embedding_size
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.regularizer_scale = regularizer_scale

        log.info('X: {}'.format(self.X.shape))
        log.info('y: {}'.format(self.y.shape))
        log.info('class_num: {}'.format(self.class_num))
        log.info('kernel_lens: {}'.format(self.kernel_lens))
        log.info('num_outputs: {}'.format(self.num_outputs))
        log.info('learning_rate: {}'.format(self.learning_rate))
        log.info('regularizer_scale: {}'.format(regularizer_scale))

        pools = []
        for k in self.kernel_lens:
            pools.append(self._get_conv_layer(k))
        cnn_output = tf.concat(pools, axis=3)
        log.info('cnn_output: {}'.format(cnn_output.shape))
        
        self.h_cnn = tf.reshape(cnn_output, shape=[-1, len(self.kernel_lens) * self.num_outputs])
        log.info('h_cnn: {}'.format(self.h_cnn.shape))

        self.h_cnn_dropout = tf.layers.dropout(self.h_cnn, 0.5)

        self.logits = tf.contrib.layers.fully_connected(
                inputs=self.h_cnn_dropout,
                num_outputs=self.class_num)
        log.info('logits: {}'.format(self.logits.shape))

        self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, 
                                                        logits=self.logits),
                name='cost'
                )
        tf.summary.scalar("cost", self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.summary_op = tf.summary.merge_all()
    
    def _get_conv_layer(self, kernel_len):
        with tf.variable_scope('kernel_{}'.format(kernel_len)) as s:
            log.info('kernel_{}'.format(kernel_len))
            cnn = tf.contrib.layers.conv2d(
                    inputs=self.X, 
                    num_outputs=self.num_outputs, 
                    kernel_size=[kernel_len, self.embedding_size], 
                    stride=1,
                    padding='VALID')
            shape = cnn.shape
            log.info('shape: {}'.format(shape))
            pool = tf.nn.max_pool(
                    value=cnn,
                    ksize=[1, shape[1], 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID')
            log.info('pool: {}'.format(pool.shape))
            return pool
