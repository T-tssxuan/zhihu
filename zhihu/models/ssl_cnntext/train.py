import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, NagtiveSamplingTopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score
from .cnntext import CNNText

summary_path = Tools.get_tf_summary_path()

log = Tools.get_logger('cnn text')
learning_rate = 0.01
batch_size = 128
topic_num = 1999
show_step = 10
test_size = 1000

log.info('begin init network')
# feed desc word representation into the network
X_word_len = 50
X_word = tf.placeholder(tf.float32, [None, X_word_len, 256], name='X_word')

# concat the input
X = X_word

y = tf.placeholder(tf.int32, [None, 1])

cnntext = CNNText(X, y, topic_num, learning_rate=learning_rate)

# init the data providers
log.info('load topic')
dp_topic = NagtiveSamplingTopicProvider()
data_topic_test = dp_topic.test(test_size, topic_num)
log.info('data_topic_test: {}'.format(data_topic_test.shape))

log.info('begin word desc data provider')
dp_word = DataProvider(DataPathConfig.get_question_train_word_topic_split_set_path(),
                            DataPathConfig.get_word_embedding_path())
log.info('begin word desc test data')
data_word_test, _ = dp_word.test(test_size, X_word_len)
log.info('data_word_test: {}'.format(len(data_word_test)))

score = Score()
log.info('begin train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    for i in range(10000000):
        data_word, _ = dp_word.next(batch_size, X_word_len)
        data_topic = dp_topic.next(batch_size, topic_num)

        feed_dict={
                   X_word: data_word,
                   y: data_topic
                  }
        # feed_dict={
        #            X_word: data_word_test,
        #            y: data_topic_test 
        #           }
        sess.run(cnntext.optimizer, feed_dict=feed_dict)
        if i % show_step == 0:
            feed_dict={
                       X_word: data_word_test,
                       y: data_topic_test 
                      }
            eval_cost, logits, summary = sess.run([cnntext.eval_cost, cnntext.logits, cnntext.summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)

            avg = data_topic.sum() / data_topic.shape[0]

            log.info('desc miss ratio: {:.4f}%'.format(dp_word.miss_ratio))
            log.info('step: {}, eval_cost: {:.6f}, offset: {}, avg: {:.4f}'.format(i, eval_cost, dp_word.offset, avg))
            _score = score.score(logits, data_topic_test)
            log.info('eval score: {}'.format(_score))
    summary_writer.close()

log.info('finished train')

