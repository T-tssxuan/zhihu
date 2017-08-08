import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, NagtiveSamplingTopicProvider, TopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score
from .cnntext import CNNText

summary_path = Tools.get_tf_summary_path()

log = Tools.get_logger('cnn text')
learning_rate = 0.1
batch_size = 256
topic_num = 2000
show_step = 50
test_size = 1000
num_sampled = 5
num_true = 3
l2_reg_lambda = 0.001

log.info('begin init network')
# feed desc word representation into the network
X_word_len = 60
X_word = tf.placeholder(tf.float32, [None, X_word_len, 256], name='X_word')

# concat the input
X = X_word

y = tf.placeholder(tf.int32, [None, num_true])

cnntext = CNNText(X, y, topic_num, learning_rate=learning_rate,
        num_sampled=num_sampled, num_true=num_true, l2_reg_lambda=l2_reg_lambda)

# init the data providers
# log.info('load topic')
# path = DataPathConfig.get_question_topic_train_topic_split_set_path()
# dp_topic = NagtiveSamplingTopicProvider(path=path, num_true=1)
#
# log.info('begin word data provider')
# dp_word = DataProvider(DataPathConfig.get_question_train_word_topic_split_set_path(),
#                             DataPathConfig.get_word_embedding_path())

log.info('load topic')
path = DataPathConfig.get_question_topic_train_set_path()
dp_topic = NagtiveSamplingTopicProvider(path=path, num_true=num_true)
data_topic_test = dp_topic.test(test_size, topic_num)
log.info('data_topic_test: {}'.format(data_topic_test.shape))

log.info('begin load word provider')
dp_word = DataProvider(DataPathConfig.get_question_train_word_set_path(),
                            DataPathConfig.get_word_embedding_path())
log.info('begin word test data')
data_word_test, _ = dp_word.test(test_size, X_word_len)
log.info('data_word_test: {}'.format(len(data_word_test)))

log.info('load topic eval')
dp_topic_eval = TopicProvider(DataPathConfig.get_question_topic_train_set_path())
data_topic_test_eval = dp_topic_eval.test(test_size, topic_num)
log.info('data_topic_test_eval: {}'.format(data_topic_test_eval.shape))


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
        _, summary = sess.run([cnntext.optimizer, cnntext.summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary, i)
        if i % show_step == 0:
            feed_dict={
                       X_word: data_word_test,
                       y: data_topic_test
                      }
            logits, eval_cost, summary = sess.run([cnntext.logits, cnntext.eval_cost, cnntext.summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)

            avg = data_topic_test.sum() / data_topic_test.shape[0]

            log.info('desc miss ratio: {:.4f}%'.format(dp_word.miss_ratio))
            log.info('step: {}, eval_cost: {:.6f}, offset: {}, avg: {:.4f}'.format(i, eval_cost, dp_word.offset, avg))
            # log.info('step: {}, offset: {}, avg: {:.4f}'.format(i, dp_word.offset, avg))
            _score = score.score(logits, data_topic_test_eval)
            log.info('eval score: {}'.format(_score))
    summary_writer.close()

log.info('finished train')

