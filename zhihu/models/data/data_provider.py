from ...config.data_path_config import DataPathConfig
import gensim
import numpy as np

total = 2999967
wv_demension = 256

class DataProvider:
    def __init__(self, data_file_path, embedding_path, is_need_length=True):
        self.offset = 0
        self.data_file_path = data_file_path
        self.end_pos = int(total * 0.7)
        self.padding_wv = [0. for i in range(wv_demension)]
        self.is_need_length = is_need_length

        self.embeding = None
        if embedding_path != '':
            self.embeding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path).wv

    def _get_data(self, f, size):
        data = []
        length = []
        for i in range(size):
            ll = f.readline()
            if ll == '':
                break
            items = ll.rstrip().split(',')
            if self.is_need_length:
                length.append(len(items))

            row = []
            if self.embeding:
                for it in items:
                    if it in self.embeding:
                        row.append(self.embeding[it])
                    else:
                        row.append(self.padding_wv)
            if self.embeding:
                data.append(row)
            else:
                data.append(items)

        return data, length

    def next(self, batch_size, max_time):
        if self.offset == 0:
            self.data_file = open(self.data_file_path, 'r')

        if self.offset + batch_size > self.end_pos:
            self.offset = self.end_pos - batch_size

        self.offset = (self.offset + batch_size) % self.end_pos

        data, length = self._get_data(self.data_file, batch_size)

        if self.offset == 0:
            self.data_file.close()

        return data, length
    
    def test(self):
        f = open(self.data_file_path, 'r')
        f.readlines(self.end_pos)
        data, length = self._get_data(f, total - self.end_pos)
        return data, length

class TopicProvider(DataProvider):
    def __init__(self, topic_file_path):
        super(TopicProvider, self).__init__(topic_file_path, '')
        self.topic_dict = dict()
        with open(DataPathConfig.get_topic_set_path(), 'r') as f:
            for idx, line in enumerate(f):
                self.topic_dict[line.rstrip()] = idx
        self.num = len(self.topic_dict.keys())

    def _one_hot(self, sentences):
        vecs = [np.zeros(self.num) for _ in range(len(sentences))]
        for idx in range(len(sentences)):
            for topic in sentences[idx]:
                vecs[idx][self.topic_dict[topic]] = 1.;
        return np.vstack(vecs)

    def next(self, batch_size):
        data, _ = super(TopicProvider, self).next(batch_size)
        return self._one_hot(data)
    
    def test(self):
        data, _ = super(TopicProvider, self).test()
        return self._one_hot(data)

if __name__ == '__main__':
    lp = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                      DataPathConfig.get_char_embedding_path())
    print(lp.next(2)[0])