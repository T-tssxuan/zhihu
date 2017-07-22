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
        self.is_need_length = is_need_length

        self.embeding = None
        if embedding_path != '':
            self.embeding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path).wv
            self.padding_wv = [0. for i in range(wv_demension)]
        else:
            self.padding_wv = 'END'

    def _get_data(self, f, size, fixed_length=0):
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

            # if need to padding to the same length
            if self.is_need_length and fixed_length > 0:
                if fixed_length > len(row):
                    row += [self.padding_wv for i in range(fixed_length - len(row))]
                else:
                    row = row[:fixed_length]

            if self.embeding:
                data.append(row)
            else:
                data.append(items)

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
        vecs = [np.zeros(self.num) for _ in range(len(sentences))]
        for idx in range(len(sentences)):
            for topic in sentences[idx]:
                vecs[idx][self.topic_dict[topic]] = 1.;
            if fixed_length > 0 and fixed_length > self.num:
                vecs[idx] += [0 for _ in range(fixed_length - self.num)]
        return vecs

    def next(self, batch_size, fixed_length):
        sentences, _ = super(TopicProvider, self).next(batch_size)
        return self._one_hot(sentences, fixed_length)
    
    def test(self, fixed_length):
        sentences, _ = super(TopicProvider, self).test()
        return self._one_hot(sentences, fixed_length)

if __name__ == '__main__':
    lp = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                      DataPathConfig.get_char_embedding_path())
    print(lp.next(2)[0])
