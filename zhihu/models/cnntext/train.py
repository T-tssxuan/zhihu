import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider, PropagatedTopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score
from .cnntext import CNNText

log = Tools.get_logger('cnn text')
learning_rate = 0.001
batch_size = 128
topic_num = 1999
show_step = 100
test_size = 10

log.info('begin init network')
# feed desc word representation into the network
X_word_desc_len = 40
X_word_desc = tf.placeholder(tf.float32, [None, X_word_desc_len, 256], name='X_word_desc')

# feed title word representation into the network
X_word_title_len = 20
X_word_title = tf.placeholder(tf.float32, [None, X_word_title_len, 256], name='X_word_title')

# concat the input
X = tf.concat([X_word_desc, X_word_title], axis=1)

y = tf.placeholder(tf.float32, [None, topic_num])

cnntext = CNNText(X, y, topic_num, learning_rate=learning_rate)

# init the data providers
log.info('load topic')
dp_topic = TopicProvider(DataPathConfig.get_question_topic_train_set_path())
data_topic_test = dp_topic.test(test_size, topic_num)
log.info('data_topic_test: {}'.format(data_topic_test.shape))

log.info('begin word desc data provider')
dp_word_desc = DataProvider(DataPathConfig.get_question_train_word_desc_set_path(),
                            DataPathConfig.get_word_embedding_path())
log.info('begin word desc test data')
data_word_desc_test, _ = dp_word_desc.test(test_size, X_word_desc_len)

log.info('begin word title init data provider')
dp_word_title = DataProvider(DataPathConfig.get_question_train_word_title_set_path(),
                             DataPathConfig.get_word_embedding_path())
log.info('begin word title test data')
data_word_title_test, _ = dp_word_title.test(test_size, X_word_title_len)


score = Score()
log.info('begin train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000000):
        data_word_desc, _ = dp_word_desc.next(batch_size, X_word_desc_len)
        data_word_title, _ = dp_word_title.next(batch_size, X_word_title_len)
        data_topic = dp_topic.next(batch_size, topic_num)

        feed_dict={
                   X_word_desc: data_word_desc,
                   X_word_title: data_word_title,
                   y: data_topic
                  }
        sess.run(cnntext.optimizer, feed_dict=feed_dict)
        if i % show_step == 0:
            feed_dict={
                       X_word_desc: data_word_desc_test,
                       X_word_title: data_word_title_test,
                       y: data_topic_test 
                      }
            cost, logits = sess.run([cnntext.cost, cnntext.logits], feed_dict=feed_dict)
            avg = data_topic.sum() / data_topic.shape[0]
            # for l in logits:
            #     print(' '.join([str(e) for e in l]))
            # log.info('logits: {}'.format(logits))
            log.info('step: {}, cost: {:.6f}, offset: {}, avg: {:.4f}'.format(i, cost, dp_word_desc.offset, avg))
            _score = score.score(logits, data_topic_test)
            log.info('eval score: {}'.format(_score))

log.info('finished train')

