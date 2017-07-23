import gensim
import numpy as np
from ...utils.tools import Tools
from gensim import models,corpora
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

log.info('tfidf_data_provider')

total = 2999967
wv_demension = 256

class DataProvider:
    def __init__(self):
        self.offset = 0
        self.end_pos = int(total * 0.7)
        self.tfidf = gensim.interfaces.TransformedCorpus.load(dir_path + './tf_idf')

    def _get_data(self, pos):
        data = []
        for i in range(size):
            j

        return data, length

    def next(self, batch_size, fixed_length=0):
        if self.offset == 0:
            self.data_file = open(self.data_file_path, 'r')

        if self.offset + batch_size > self.end_pos:
            self.offset = self.end_pos - batch_size

        self.offset = (self.offset + batch_size) % self.end_pos

        data, length = self._get_data(self.data_file, batch_size, fixed_length)

        if self.offset == 0:
            self.data_file.close()

        return data, length
    
    def test(self, fixed_length):
        f = open(self.data_file_path, 'r')
        f.readlines(self.end_pos)
        data, length = self._get_data(f, total - self.end_pos, fixed_length)
        return data, length

class TopicProvider(DataProvider):
    def __init__(self, topic_file_path):
        super(TopicProvider, self).__init__(topic_file_path, '', False)
        self.topic_dict = dict() 
        with open(DataPathConfig.get_topic_set_path(), 'r') as f:
            for idx, line in enumerate(f):
                self.topic_dict[line.rstrip()] = idx
        self.num = len(self.topic_dict.keys())

    def _one_hot(self, sentences, fixed_length=0):
        length = max(self.num, fixed_length)
        vecs = np.zeros((len(sentences), length))
        for idx in range(len(sentences)):
            for topic in sentences[idx]:
                vecs[idx][self.topic_dict[topic]] = 1.;
        return vecs

    def next(self, batch_size, fixed_length=0):
        sentences, _ = super(TopicProvider, self).next(batch_size)
        return self._one_hot(sentences, fixed_length)
    
    def test(self, fixed_length=0):
        sentences, _ = super(TopicProvider, self).test()
        return self._one_hot(sentences, fixed_length)

class BinaryTopicProvider(TopicProvider):
    def __init__(self, topic_file_path):
        super(BinaryTopicProvider, self).__init__(topic_file_path)

    def _to_binary(self, vecs, class_idx):
        vecs = vecs[:, class_idx].reshape(-1, 1)
        vecs = np.hstack([np.zeros((vecs.shape[0], 1)), vecs])
        return vecs

    def next(self, batch_size, class_idx):
        vecs = super(BinaryTopicProvider, self).next(batch_size)
        return self._to_binary(vecs, class_idx)
    
    def test(self, class_idx):
        vecs = super(BinaryTopicProvider, self).test()
        return self._to_binary(vecs, class_idx)

if __name__ == '__main__':
    lp = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                      DataPathConfig.get_char_embedding_path())
    print(lp.next(2)[0])
