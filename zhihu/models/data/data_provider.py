from ...config.data_path_config import DataPathConfig
import gensim
import numpy as np

total = 2999967
wv_demension = 256

class DataProvider:
    def __init__(self, data_file_path, embedding_path):
        self.offset = 0
        self.data_file_path = data_file_path
        self.end_pos = int(total * 0.7)
        self.padding_wv = [0. for i in range(wv_demension)]

        self.embeding = None
        if embedding_path != '':
            self.embeding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path).wv

    def _get_data(self, f, size):
        data = []
        for i in range(size):
            ll = f.readline()
            print(ll)
            if ll == '':
                break
            items = ll.rstrip().split(',')
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

        return data

    def next(self, batch_size):
        if self.offset == 0:
            self.data_file = open(self.data_file_path, 'r')

        if self.offset + batch_size > self.end_pos:
            self.offset = self.end_pos - batch_size

        self.offset = (self.offset + batch_size) % batch_size

        data = self._get_data(self.data_file, batch_size)

        if self.offset == 0:
            self.data_file.close()

        return data
    
    def train(self):
        f = open(self.data_file_path, 'r')
        f.readlines(self.end_pos)
        data = self._get_data(f, total - self.end_pos)
        return data

if __name__ == '__main__':
    lp = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                      DataPathConfig.get_char_embedding_path())
    print(lp.next(2))
