import tensorflow as tf
import numpy as np
from ...utils.tools import Tools

log = Tools.get_logger('CNNText')

class CNNText:
    def __init__(self, X, y, 
            class_num=2000, 
            kernel_lens=[3, 4, 5, 6], 
            num_outputs=512, 
            embedding_size=256,
            learning_rate=0.01,
            regularizer_scale=0.1,
            num_sampled=5,
            num_true=3,
            l2_reg_lambda=0.01):
        self.X = tf.expand_dims(X, -1)
        self.y = y
        self.class_num = class_num
        self.kernel_lens = kernel_lens
        self.embedding_size = embedding_size
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.regularizer_scale = regularizer_scale
        self.l2_reg_lambda = l2_reg_lambda

        log.info('X: {}'.format(self.X.shape))
        log.info('y: {}'.format(self.y.shape))
        log.info('class_num: {}'.format(self.class_num))
        log.info('kernel_lens: {}'.format(self.kernel_lens))
        log.info('num_outputs: {}'.format(self.num_outputs))
        log.info('learning_rate: {}'.format(self.learning_rate))
        log.info('regularizer_scale: {}'.format(self.regularizer_scale))
        log.info('num_sampled: {}'.format(num_sampled))
        log.info('l2_reg_lambda: {}'.format(self.l2_reg_lambda))

        self.l2_loss = tf.constant(0.0)

        pools = []
        for k in self.kernel_lens:
            pools.append(self._get_conv_layer(k))
        cnn_output = tf.concat(pools, axis=3)
        log.info('cnn_output: {}'.format(cnn_output.shape))
        
        self.h_cnn = tf.reshape(cnn_output, shape=[-1, len(self.kernel_lens) * self.num_outputs])
        log.info('h_cnn: {}'.format(self.h_cnn.shape))

        self.h_cnn_dropout = tf.layers.dropout(self.h_cnn, 0.5)


        softmax_w = tf.Variable(tf.truncated_normal((class_num, self.h_cnn_dropout.shape[1].value)), name='softmax_weight')
        softmax_b = tf.Variable(tf.zeros(class_num), name="softmax_bias") 
        log.info('softmax_w: {}'.format(softmax_w.shape))
        log.info('softmax_b: {}'.format(softmax_b.shape))

        # the train graph
        log.info('init the softmax sampling sub-graph')
        self.ns_train_cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=softmax_w,
                biases=softmax_b,
                labels=y,
                inputs=self.h_cnn_dropout,
                num_sampled=num_sampled,
                num_classes=class_num,
                num_true=num_true,
                name='ns_train_loss'))
        # self.train_cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        #         weights=softmax_w,
        #         biases=softmax_b,
        #         labels=y,
        #         inputs=self.h_cnn_dropout,
        #         num_sampled=num_sampled,
        #         num_classes=class_num,
        #         num_true=1,
        #         name='train_loss'))
        log.info('ns_train_cost: {}'.format(self.ns_train_cost))
        tf.summary.scalar("ns_train_cost", self.ns_train_cost)

        self.l2_loss += tf.nn.l2_loss(softmax_w)
        self.l2_loss += tf.nn.l2_loss(softmax_b)
        tf.summary.scalar("l2_loss", self.l2_loss)

        self.train_cost = self.ns_train_cost + self.l2_reg_lambda * self.l2_loss
        tf.summary.scalar("train_cost", self.train_cost)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.train_cost)
        log.info('finihsed init softmax sampling sub-graph')

        # the eval sub-graph
        log.info('init the eval sub-graph')
        logits = tf.matmul(self.h_cnn_dropout, tf.transpose(softmax_w))
        self.logits = tf.nn.bias_add(logits, softmax_b)
        log.info('logits: {}'.format(self.logits.shape))
        logits_mean = tf.reduce_mean(self.logits)
        tf.summary.scalar("logits_mean", logits_mean)

        labels_one_hot = tf.reduce_sum(tf.one_hot(y, class_num), axis=1)
        # labels_one_hot = tf.one_hot(y, class_num)
        log.info('labels_one_hot: {}'.format(labels_one_hot.shape))

        self.eval_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits))
        log.info('eval_cost: {}'.format(self.eval_cost))
        tf.summary.scalar("eval_cost", self.eval_cost)
        log.info('finished init ssl cnntext')

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
