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
from ..data.data_provider import DataProvider, TopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools

log = Tools.get_logger('dynamic_rnn')

learning_rate = 0.01
batch_size = 128
topic_num = 2048

def get_dynamic_rnn_graph(X, X_length, scope):
    with tf.variable_scope(scope):
        cell = tf.contrib.rnn.LSTMCell(num_units=64, state_is_tuple=True)
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                 dtype=tf.float64,
                                                 sequence_length=X_length,
                                                 inputs=X)
    return last_states.h

log.info('begin init network')
# feed desc word representation into the network
X_word_desc_max_time = 60
X_word_desc = tf.placeholder(tf.float64, [batch_size, X_word_desc_max_time, 256], name='X_word_desc')
X_word_desc_length = tf.placeholder(tf.int32, [batch_size], name='X_word_desc_length')
word_desc_lstm = get_dynamic_rnn_graph(X_word_desc, X_word_desc_length, 'word_desc')
log.info('word_desc_lstm({})'.format(word_desc_lstm.shape))

# feed title word representation into the network
X_word_title_max_time = 20
X_word_title = tf.placeholder(tf.float64, [batch_size, X_word_title_max_time, 256], name='X_word_title')
X_word_title_length = tf.placeholder(tf.int32, [batch_size], name='X_word_title_length')
word_title_lstm = get_dynamic_rnn_graph(X_word_title, X_word_title_length, 'word_title')
log.info('word_title_lstm({})'.format(word_title_lstm.shape))

# feed desc char representation into the network
X_char_desc_max_time = 200
X_char_desc = tf.placeholder(tf.float64, [batch_size, X_char_desc_max_time, 256], name='X_char_desc')
X_char_desc_length = tf.placeholder(tf.int32, [batch_size], name='X_char_desc_length')
char_desc_lstm = get_dynamic_rnn_graph(X_char_desc, X_char_desc_length, 'char_desc')
log.info('char_desc_lstm({})'.format(char_desc_lstm.shape))

# feed title char representation into the network
X_char_title_max_time = 25
X_char_title = tf.placeholder(tf.float64, [batch_size, X_char_title_max_time, 256], name='X_char_title')
X_char_title_length = tf.placeholder(tf.int32, [batch_size], name='X_char_title_length')
char_title_lstm = get_dynamic_rnn_graph(X_char_title, X_char_title_length, 'char_title')
log.info('char_title_lstm({})'.format(char_title_lstm.shape))

# the topic placeholder
topics = tf.placeholder(tf.float64, [batch_size, topic_num])

lstm = tf.concat([word_desc_lstm, word_title_lstm, char_desc_lstm, char_title_lstm], axis=1)
# lstm = char_title_lstm

log.info('lstm({})'.format(lstm.shape))
fc1 = tf.contrib.layers.fully_connected(inputs=lstm, num_outputs=512)
drop_fc1 = tf.nn.dropout(fc1, 0.5)
fc2 = tf.contrib.layers.fully_connected(inputs=drop_fc1, num_outputs=1024)
drop_fc2 = tf.nn.dropout(fc2, 0.5)
logits = tf.contrib.layers.fully_connected(inputs=drop_fc2, num_outputs=topic_num)

values, indices = tf.nn.top_k(logits, 5)

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
log.info('begin char desc init data provider')
dp_char_desc = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                            DataPathConfig.get_char_embedding_path())
log.info('begin char title init data provider')
dp_char_title = DataProvider(DataPathConfig.get_question_train_character_title_set_path(),
                             DataPathConfig.get_char_embedding_path())
log.info('begin topic init data provider')
dp_topic = TopicProvider(DataPathConfig.get_question_topic_train_set_path())

log.info('begin train')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        data_word_desc, data_word_desc_length = dp_word_desc.next(batch_size, X_word_desc_max_time)
        data_word_title, data_word_title_length = dp_word_title.next(batch_size, X_word_title_max_time)
        data_char_desc, data_char_desc_length = dp_char_desc.next(batch_size, X_char_desc_max_time)
        data_char_title, data_char_title_length = dp_char_title.next(batch_size, X_char_title_max_time)
        data_topic = dp_topic.next(batch_size, topic_num)

        feed_dict={
                   X_word_desc: data_word_desc,
                   X_word_desc_length: data_word_desc_length,
                   X_word_title: data_word_title,
                   X_word_title_length: data_word_title_length,
                   X_char_desc: data_char_desc,
                   X_char_desc_length: data_char_desc_length,
                   X_char_title: data_char_title,
                   X_char_title_length: data_char_title_length,
                   topics: data_topic
                  }
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 10 == 0:
            loss = sess.run(cost, feed_dict=feed_dict)
            tp = np.array(data_topic)
            avg = tp.sum() / tp.shape[0]
            log.info('step: {}, loss: {:.6f}, offset: {}, avg: {:.4f}'.format(i, loss, dp_char_title.offset, avg))

log.info('finished train')
