# Basic lstm

import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score
from gensim import models,corpora
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

log = Tools.get_logger('dynamic_rnn')

topic_num = 1000
documents = []
with open(DataPathConfig.get_question_train_word_desc_set_path(), 'r') as f:
    for line in f:
        documents.append(line.rstrip().split(','))

with open(DataPathConfig.get_question_train_word_title_set_path(), 'r') as f:
    for idx, line in enumerate(f):
        documents[idx] += line.rstrip().split(',')

dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

corpus_tfidf.save(dir_path + '/tf_idf')






log.info('lstm({})'.format(lstm.shape))
fc1 = tf.contrib.layers.fully_connected(inputs=lstm, num_outputs=1024)
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
log.info('begin char desc init data provider')
dp_char_desc = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                            DataPathConfig.get_char_embedding_path())
log.info('begin char title init data provider')
dp_char_title = DataProvider(DataPathConfig.get_question_train_character_title_set_path(),
                             DataPathConfig.get_char_embedding_path())
log.info('begin topic init data provider')
dp_topic = TopicProvider(DataPathConfig.get_question_topic_train_set_path())

score = Score()
log.info('begin train')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000000):
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
            loss, _lstm_mean, _logits_mean, _drop_fc1_mean, _logits = sess.run([cost, lstm_mean, logits_mean, drop_fc1_mean, logits], feed_dict=feed_dict)
            # log.info('lstm_mean: {}'.format(_lstm_mean))
            # log.info('drop_fc1_mean: {}'.format(_drop_fc1_mean))
            log.info('logits_mean: {}'.format(_logits_mean))
            avg = data_topic.sum() / data_topic.shape[0]
            log.info('step: {}, loss: {:.6f}, offset: {}, avg: {:.4f}'.format(i, loss, dp_char_title.offset, avg))
            _score = score.score(_logits, data_topic)
            log.info('eval score: {}'.format(_score))

log.info('finished train')
