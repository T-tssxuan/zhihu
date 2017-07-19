# 2017/7/18
"""
Format and validate the question and question topic file
"""
import os
from time import time
from ..config.data_path_config import DataPathConfig, RawDataPathConfig
from ..utils.tools import Tools

def format_question_data():
    q_in = open(RawDataPathConfig.get_question_train_set_path(), 'r')
    qt_in = open(RawDataPathConfig.get_question_topic_train_set_path(), 'r')

    q_w_title_out = open(DataPathConfig.get_question_train_word_title_set_path(), 'w')
    q_w_desc_out = open(DataPathConfig.get_question_train_word_desc_set_path(), 'w')
    q_c_title_out = open(DataPathConfig.get_question_train_character_title_set_path(), 'w')
    q_c_desc_out = open(DataPathConfig.get_question_train_character_desc_set_path(), 'w')
    qt_out = open(DataPathConfig.get_question_topic_train_set_path(), 'w')

    missed_match_count = 0

    t0 = time()
    num = 0

    while True:
        num += 1
        if num % 100000 == 0:
            print('finished: {} in {:.2f}s'.format(num, time() - t0))
        q_l = q_in.readline()
        qt_l = qt_in.readline()
        if q_l == '' and qt_l == '':
            break

        q_items = q_l.rstrip('\n').split('\t')
        qt_items = qt_l.rstrip('\n').split('\t')

        if q_items[0] != qt_items[0]:
            missed_match_count += 1
            continue

        q_c_title_out.write(q_items[1] + '\n')
        q_c_desc_out.write(q_items[3] + '\n')
        q_w_title_out.write(q_items[2] + '\n')
        q_w_desc_out.write(q_items[4] + '\n')

        qt_out.write(qt_items[1] + '\n')

    q_in.close()
    qt_in.close()

    q_w_title_out.close()
    q_w_desc_out.close()
    q_c_title_out.close()
    q_c_desc_out.close()
    qt_out.close()

    print('finished all {} in {:.2f}s, missed: {}'.format(num, time() - t0, missed_match_count))

if __name__ == '__main__':
    format_question_data()
