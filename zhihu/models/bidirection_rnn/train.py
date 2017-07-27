# Basic lstm

'''
1. feed the word embedding into the lstm network get: lstmw
2. feed the character embedding into the lstm network get: lstmc
3. concat the two vector get: [lstmw, lstmw] => vec
4. feed the vec to fc neetwork and a softmax layer
5. using the loss = cross_entropy to get the result
6. every time using the top 5 as the predict result
'''

import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider, PropagatedTopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score

log = Tools.get_logger('dynamic_rnn')

learning_rate = 0.0001
batch_size = 128
topic_num = 1999
num_hidden = 512

def get_dynamic_rnn_graph(X, X_length, scope):
    with tf.variable_scope(scope):
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=X_length,
                inputs=X)
    return tf.concat([last_states[0].h, last_states[0].h], axis=1)

log.info('begin init network')
# feed desc word representation into the network
X_word_desc_max_time = 60
X_word_desc = tf.placeholder(tf.float32, [None, X_word_desc_max_time, 256], name='X_word_desc')
X_word_desc_length = tf.placeholder(tf.int32, [None], name='X_word_desc_length')
word_desc_lstm = get_dynamic_rnn_graph(X_word_desc, X_word_desc_length, 'word_desc')
log.info('word_desc_lstm({})'.format(word_desc_lstm.shape))

# feed title word representation into the network
X_word_title_max_time = 20
X_word_title = tf.placeholder(tf.float32, [None, X_word_title_max_time, 256], name='X_word_title')
X_word_title_length = tf.placeholder(tf.int32, [None], name='X_word_title_length')
word_title_lstm = get_dynamic_rnn_graph(X_word_title, X_word_title_length, 'word_title')
log.info('word_title_lstm({})'.format(word_title_lstm.shape))

# # feed desc char representation into the network
# X_char_desc_max_time = 200
# X_char_desc = tf.placeholder(tf.float32, [None, X_char_desc_max_time, 256], name='X_char_desc')
# X_char_desc_length = tf.placeholder(tf.int32, [None], name='X_char_desc_length')
# char_desc_lstm = get_dynamic_rnn_graph(X_char_desc, X_char_desc_length, 'char_desc')
# log.info('char_desc_lstm({})'.format(char_desc_lstm.shape))
#
# # feed title char representation into the network
# X_char_title_max_time = 25
# X_char_title = tf.placeholder(tf.float32, [None, X_char_title_max_time, 256], name='X_char_title')
# X_char_title_length = tf.placeholder(tf.int32, [None], name='X_char_title_length')
# char_title_lstm = get_dynamic_rnn_graph(X_char_title, X_char_title_length, 'char_title')
# log.info('char_title_lstm({})'.format(char_title_lstm.shape))

# the topic placeholder
topics = tf.placeholder(tf.float32, [None, topic_num])

# lstm = tf.concat([word_desc_lstm, word_title_lstm, char_desc_lstm, char_title_lstm], axis=1)
lstm = tf.concat([word_desc_lstm, word_title_lstm], axis=1)
lstm_mean = tf.reduce_mean(lstm)

log.info('lstm({})'.format(lstm.shape))
fc1 = tf.contrib.layers.fully_connected(inputs=lstm, num_outputs=4*num_hidden)
log.info('fc1: {}'.format(fc1.shape))
drop_fc1 = tf.nn.dropout(fc1, 0.5)
log.info('drop_fc1: {}'.format(drop_fc1.shape))
drop_fc1_mean = tf.reduce_mean(drop_fc1)
# fc2 = tf.contrib.layers.fully_connected(inputs=drop_fc1, num_outputs=1024)
# drop_fc2 = tf.nn.dropout(fc2, 0.5)
# log.info('drop_fc2: {}'.format(drop_fc2.shape))
logits = tf.contrib.layers.fully_connected(inputs=drop_fc1, num_outputs=topic_num)
logits_mean = tf.reduce_mean(logits)

top_k_values, top_k_indices = tf.nn.top_k(logits, 5)

log.info('logits: {}'.format(logits.shape))
log.info('topics: {}'.format(topics.shape))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=topics, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# init the data providers
log.info('begin word desc data provider')
dp_word_desc = DataProvider(DataPathConfig.get_question_train_word_desc_set_path(),
                            DataPathConfig.get_word_embedding_path())
log.info('begin word title init data provider')
dp_word_title = DataProvider(DataPathConfig.get_question_train_word_title_set_path(),
                             DataPathConfig.get_word_embedding_path())
# log.info('begin char desc init data provider')
# dp_char_desc = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
#                             DataPathConfig.get_char_embedding_path())
# log.info('begin char title init data provider')
# dp_char_title = DataProvider(DataPathConfig.get_question_train_character_title_set_path(),
#                              DataPathConfig.get_char_embedding_path())
# log.info('begin topic init data provider')
dp_topic = TopicProvider(DataPathConfig.get_question_topic_train_set_path())
# dp_topic = PropagatedTopicProvider()

log.info('begin load test data')
test_word_desc, test_word_desc_length = dp_word_desc.test()
test_word_title, test_word_title_length = dp_word_title.test()
# test_char_desc, test_char_desc_length = dp_char_desc.test()
# test_char_title, test_char_title_length = dp_char_title.test()
test_topic = dp_topic.test()
log.info('finished load test data')

score = Score()
log.info('begin train')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000000):
        data_word_desc, data_word_desc_length = dp_word_desc.next(batch_size, X_word_desc_max_time)
        data_word_title, data_word_title_length = dp_word_title.next(batch_size, X_word_title_max_time)
        # data_char_desc, data_char_desc_length = dp_char_desc.next(batch_size, X_char_desc_max_time)
        # data_char_title, data_char_title_length = dp_char_title.next(batch_size, X_char_title_max_time)
        data_topic = dp_topic.next(batch_size, topic_num)

        feed_dict={
                   X_word_desc: data_word_desc,
                   X_word_desc_length: data_word_desc_length,
                   X_word_title: data_word_title,
                   X_word_title_length: data_word_title_length,
                   # X_char_desc: data_char_desc,
                   # X_char_desc_length: data_char_desc_length,
                   # X_char_title: data_char_title,
                   # X_char_title_length: data_char_title_length,
                   topics: data_topic
                  }
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 10 == 0:
            loss, _lstm_mean, _logits_mean, _drop_fc1_mean, _logits = sess.run([cost, lstm_mean, logits_mean, drop_fc1_mean, logits], feed_dict=feed_dict)
            # log.info('lstm_mean: {}'.format(_lstm_mean))
            # log.info('drop_fc1_mean: {}'.format(_drop_fc1_mean))
            log.info('logits_mean: {}'.format(_logits_mean))
            avg = data_topic.sum() / data_topic.shape[0]
            log.info('step: {}, loss: {:.6f}, offset: {}, avg: {:.4f}'.format(i, loss, dp_word_desc.offset, avg))
            _score = score.score(_logits, data_topic)
            log.info('eval score: {}'.format(_score))

        if i % 100 == 0:
            feed_dict = {
                    X_word_desc: test_word_desc,
                    X_word_desc_length: test_word_desc_length,
                    X_word_title: test_word_title,
                    X_word_title_length: test_word_title_length,
                    # X_char_desc: test_char_desc,
                    # X_char_desc_length: test_char_desc_length,
                    # X_char_title: test_char_title,
                    # X_char_title_length: test_char_title_length,
                    topics: test_topic 
                    }
            top_k = sess.run(top_k_indices, feed_dict={feed_dict})
            np.save('./preditc', top_k)

log.info('finished train')
