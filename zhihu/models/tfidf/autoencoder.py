import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider, TfidfDataProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score
from gensim import models,corpora
import os 

learning_rate = 0.001
batch_size = 64
hidden_size = 5000

log = Tools.get_logger('autoencoder')

log.info('init autoencoder')
X = tf.placeholder(tf.float32, [None, 50000])

efc1 = tf.contrib.layers.fully_connected(inputs=X, num_outputs=20000)
efc2 = tf.contrib.layers.fully_connected(inputs=efc1, num_outputs=5000)

dfc1 = tf.contrib.layers.fully_connected(inputs=efc2, num_outputs=20000)
dfc2 = tf.contrib.layers.fully_connected(inputs=dfc1, num_outputs=50000)

y_pred = dfc2
y_true = X

cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

log.info('init data provider')
dp = TfidfDataProvider()

with tf.Session() as sess:
    sess.run(init)
    log.info('begin train')
    for idx in range(1000000):
        feed_dict = {X: dp.next(batch_size)}
        sess.run(optimizer, feed_dict=feed_dict)
        if idx % 10 == 0:
            loss = sess.run(cost, feed_dict=feed_dict)
            log.info('step: {}, loss: {:.4f}'.format(loss))
    log.info('finished trian')
    feed_dict = {X: dp.test()}
    loss = sess.run(cost, feed_dict)
    log.info('test loss: {:.4f}'.format(loss))
