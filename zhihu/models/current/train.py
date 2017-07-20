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
from ..data.data_provider import DataProvider, TopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools

log = Tools.get_logger('dynamic_rnn')

learning_rate = 0.01
<<<<<<< HEAD
batch_size = 1
=======
batch_size = 128
>>>>>>> 49be119e730a567e58e8e88a7cfa0d9809ebabcf

def get_dynamic_rnn_graph(X, X_length, scope):
    with tf.variable_scope(scope):
        cell = tf.contrib.rnn.LSTMCell(num_units=64, state_is_tuple=True)
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell,
                                                 dtype=tf.float64,
                                                 sequence_length=X_length,
                                                 inputs=X)
<<<<<<< HEAD
        print(tf.shape(outputs), tf.shape(last_states))
    return last_states

log.info('begin init network')
# # feed desc word representation into the network
# X_word_desc = tf.placeholder(tf.float64, [batch_size, None, 256])
# X_word_desc_length = tf.placeholder(tf.int32, [batch_size])
# word_desc_lstm = get_dynamic_rnn_graph(X_word_desc, X_word_desc_length, 'word_desc')
#
# # feed title word representation into the network
# X_word_title = tf.placeholder(tf.float64, [batch_size, None, 256])
# X_word_title_length = tf.placeholder(tf.int32, [batch_size])
# word_title_lstm = get_dynamic_rnn_graph(X_word_title, X_word_title_length, 'word_title')
#
# # feed desc char representation into the network
# X_char_desc = tf.placeholder(tf.float64, [batch_size, None, 256])
# X_char_desc_length = tf.placeholder(tf.int32, [batch_size])
# char_desc_lstm = get_dynamic_rnn_graph(X_char_desc, X_char_desc_length, 'char_desc')
=======
    return last_states

log.info('begin init network')
# feed desc word representation into the network
X_word_desc = tf.placeholder(tf.float64, [batch_size, None, 256])
X_word_desc_length = tf.placeholder(tf.int32, [batch_size])
word_desc_lstm = get_dynamic_rnn_graph(X_word_desc, X_word_desc_length, 'word_desc')

# feed title word representation into the network
X_word_title = tf.placeholder(tf.float64, [batch_size, None, 256])
X_word_title_length = tf.placeholder(tf.int32, [batch_size])
word_title_lstm = get_dynamic_rnn_graph(X_word_title, X_word_title_length, 'word_title')

# feed desc char representation into the network
X_char_desc = tf.placeholder(tf.float64, [batch_size, None, 256])
X_char_desc_length = tf.placeholder(tf.int32, [batch_size])
char_desc_lstm = get_dynamic_rnn_graph(X_char_desc, X_char_desc_length, 'char_desc')
>>>>>>> 49be119e730a567e58e8e88a7cfa0d9809ebabcf

# feed title char representation into the network
X_char_title = tf.placeholder(tf.float64, [batch_size, None, 256])
X_char_title_length = tf.placeholder(tf.int32, [batch_size])
char_title_lstm = get_dynamic_rnn_graph(X_char_title, X_char_title_length, 'char_title')

# the topic placeholder
topics = tf.placeholder(tf.float64, [batch_size, 1999])

# lstm = tf.concat([word_desc_lstm, word_title_lstm, char_desc_lstm, char_title_lstm], axis=1)
lstm = char_title_lstm

print(tf.shape(lstm))

fc1 = tf.contrib.layers.fully_connected(inputs=lstm, num_outputs=5000)
logits = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1999)

values, indices = tf.nn.top_k(logits, 5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=topics, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# init the data providers
# log.info('begin word desc data provider')
# dp_word_desc = DataProvider(DataPathConfig.get_question_train_word_desc_set_path(),
#                             DataPathConfig.get_word_embedding_path())
# log.info('begin word title init data provider')
# dp_word_title = DataProvider(DataPathConfig.get_question_train_word_title_set_path(),
#                              DataPathConfig.get_word_embedding_path())
# log.info('begin char desc init data provider')
# dp_char_desc = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
#                             DataPathConfig.get_char_embedding_path())
log.info('begin char title init data provider')
dp_char_title = DataProvider(DataPathConfig.get_question_train_character_title_set_path(),
                             DataPathConfig.get_char_embedding_path())
log.info('begin topic init data provider')
dp_topic = TopicProvider(DataPathConfig.get_question_topic_train_set_path())

log.info('begin train')
with tf.Session() as sess:
    for i in range(5001):
        data_word_desc, data_word_desc_length = dp_word_desc.next(batch_size)
        data_word_title, data_word_title_length = dp_word_title.next(batch_size)
        data_char_desc, data_char_desc_length = dp_char_desc.next(batch_size)
        data_char_title, data_cahr_title_length = dp_char_title.next(batch_size)
        data_topic = dp_topic.next(batch_size)
        print(data_topic.shape)

        feed_dict={
                   # X_word_desc: data_word_desc,
                   # X_word_desc_length: data_word_desc_length,
                   # X_word_title: data_word_title,
                   # X_word_title_length: data_word_title_length,
                   # X_char_desc: data_char_desc,
                   # X_char_desc_length: data_char_desc_length,
                   X_char_title: data_char_title,
                   X_char_title_length: data_cahr_title_length,
                   topics: data_topic
                  }
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 10 == 0:
            loss = sess.run(cost, feed_dict=feed_dict)
            log.info('step: {}, loss: {}'.format(i, loss))

log.info('finished train')
