# Basic lstm

from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from gensim import models,corpora
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

log = Tools.get_logger('generate_tfidf_vector')

documents = []

log.info('load word desc')
with open(DataPathConfig.get_question_train_word_desc_set_path(), 'r') as f:
    for line in f:
        documents.append(line.rstrip().split(','))

log.info('load word title')
with open(DataPathConfig.get_question_train_word_title_set_path(), 'r') as f:
    for idx, line in enumerate(f):
        documents[idx] += line.rstrip().split(',')

# with open('/home/luoxuan/workspace/zhihu/tmp', 'r') as f:
#     for line in f:
#         documents.append(line.rstrip().split(','))

log.info('train dictionary')
dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(keep_n=50000)

log.info('transform document')
corpus = [dictionary.doc2bow(text) for text in documents]

log.info('train models')
tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

f = open(dir_path + '/tfidf.vec', 'w')

log.info('save file')
for i in range(len(corpus)):
    vec = [0 for _ in range(50000)]
    for ele in corpus_tfidf[i]:
        vec[ele[0]] = ele[1]
    vec = [str(val) for val in vec]
    f.write(','.join(vec) + '\n')
    if i % 100000 == 0:
        log.info('finshed {:.2f}%'.format(i / len(corpus)))
