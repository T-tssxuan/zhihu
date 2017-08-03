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
        self._miss_count = 0.
        self._total_count = 0.
        self.miss_ratio = 0.

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
                    self._total_count += 1
                    if it in self.embeding:
                        row.append(self.embeding[it])
                    else:
                        row.append(self.padding_wv)
                        self._miss_count += 1

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

        self.miss_ratio = self._miss_count / max(1., self._total_count) * 100
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
    
    def test(self, size=0, fixed_length=0):
        f = open(self.data_file_path, 'r')
        f.readlines(self.end_pos)
        if size == 0:
            size = total - self.end_pos
        data, length = self._get_data(f, size, fixed_length)
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
    
    def test(self, size=0, fixed_length=0):
        sentences, _ = super(TopicProvider, self).test(
                size=size, 
                fixed_length=fixed_length)
        return self._one_hot(sentences, fixed_length)

class PropagatedTopicProvider(DataProvider):
    def __init__(self):
        topic_file_path = DataPathConfig.get_topic_with_parent_propagate()
        self.num = 1999
        super(PropagatedTopicProvider, self).__init__(topic_file_path, '', False)

    def _one_hot(self, sentences, fixed_length=0):
        length = max(self.num, fixed_length)
        vecs = np.zeros((len(sentences), length))
        for row in range(len(sentences)):
            for col in range(length):
                if sentences[row][col] == '1':
                    vecs[row][col] = 1.
        return vecs

    def next(self, batch_size, fixed_length=0):
        sentences, _ = super(PropagatedTopicProvider, self).next(batch_size)
        return self._one_hot(sentences, fixed_length)
    
    def test(self, size=0, fixed_length=0):
        sentences, _ = super(PropagatedTopicProvider, self).test(
                size=size, fixed_length=fixed_length)
        return self._one_hot(sentences, fixed_length)

class BinaryTopicProvider(PropagatedTopicProvider):
    def __init__(self):
        super(BinaryTopicProvider, self).__init__()

    def _to_binary(self, vecs, class_idx):
        vecs = vecs[:, class_idx]
        tmp = np.zeros((vecs.shape[0], 2))
        tmp[:, 0] += 1 - vecs
        tmp[:, 1] += vecs
        return tmp

    def next(self, batch_size, class_idx):
        vecs = super(BinaryTopicProvider, self).next(batch_size)
        return self._to_binary(vecs, class_idx)
    
    def test(self, class_idx, size=0, fixed_length=0):
        vecs = super(BinaryTopicProvider, self).test(
                size=size, fixed_length=fixed_length)
        return self._to_binary(vecs, class_idx)

class TfidfDataProvider(DataProvider):
    def __init__(self):
        topic_file_path = DataPathConfig.get_question_tfidf_vec_path()
        super(TfidfDataProvider, self).__init__(topic_file_path, '', False)

    def _transform(self, vecs):
        vecs = [[float(ele) for ele in vec] for vec in vecs]
        return vecs

    def next(self, batch_size):
        vecs, _ = super(TfidfDataProvider, self).next(batch_size)
        return self._transform(vecs)
    
    def test(self, size=0, fixed_length=0):
        vecs, _ = super(TfidfDataProvider, self).test(
                size=size, fixed_length=fixed_length)
        return self._transform(vecs)

if __name__ == '__main__':
    lp = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                      DataPathConfig.get_char_embedding_path())
    print(lp.next(2)[0])
