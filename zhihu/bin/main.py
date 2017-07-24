from ..preprocess.get_topic_relation import get_topic_relation
from ..preprocess.question_format import format_question_data
from ..models.data.data_provider import TopicProvider, DataProvider, BinaryTopicProvider, PropagatedTopicProvider
from ..config.data_path_config import RawDataPathConfig, DataPathConfig
import os
import sys

if __name__ == '__main__':
    if sys.argv[1] == 'preprocess_topic' or sys.argv[1] == 'all':
        get_topic_relation()
    if sys.argv[1] == 'preprocess_question' or sys.argv[1] == 'all':
        format_question_data()
    if sys.argv[1] == 'char_provider' or sys.argv[1] == 'all':
        dp = DataProvider(DataPathConfig.get_question_train_character_desc_set_path(),
                          DataPathConfig.get_char_embedding_path())
        data, length  = dp.next(2, 1000)
        print(len(data[0]), len(data[1]))
        print(length)
        print(data)
    if sys.argv[1] == 'word_provider' or sys.argv[1] == 'all':
        dp = DataProvider(DataPathConfig.get_question_train_word_title_set_path(),
                          '')
        data, length = dp.next(2)
        print(len(data[0]), len(data[1]))
        print(length)
        print(data)
    if sys.argv[1] == 'topic_provider' or sys.argv[1] == 'all':
        tp = TopicProvider(DataPathConfig.get_question_topic_train_set_path())
        for i in range(10):
            data = tp.next(2, 2048)
            print(len(data[0]), len(data[1]))
            print(data[0].sum(), data[1].sum())
            print(data.sum() / data.shape[0])
            print(data)
    if sys.argv[1] == 'binary_provider' or sys.argv[1] == 'all':
        tp = BinaryTopicProvider(DataPathConfig.get_question_topic_train_set_path())
        for i in range(100):
            for j in range(1999):
                data = tp.next(10, j)
                print(data[0].sum(), data[1].sum())
                print(data.sum() / data.shape[0])
                # print(data)
    if sys.argv[1] == 'ptp' or sys.argv[1] == 'all':
        ptp = PropagatedTopicProvider()
        for i in range(10):
            data = ptp.next(2)
            print(len(data[0]), len(data[1]))
            print(data[0].sum(), data[1].sum())
            print(data.sum() / data.shape[0])
            print(data)
