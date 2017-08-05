# 2017/7/18
"""
Format and validate the question and question topic file
"""
import os
from time import time
from ..config.data_path_config import DataPathConfig, RawDataPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('question_format')

def format_topic_idx():
    log.info('begin idx_topic')
    with open(DataPathConfig.get_topic_set_path(), 'r') as f:
        dd = dict([(key.rstrip(), str(value)) for value, key in enumerate(f)])

    with open(DataPathConfig.get_topic_idx_path(), 'w') as out:
        with open(DataPathConfig.get_question_topic_train_set_path(), 'r') as f:
            for num, line in enumerate(f):
                tmp = [dd[ele] for ele in line.rstrip().split(',')]
                out.write(','.join(tmp) + '\n')
                if num % 300000 == 0:
                    log.info('finished: {}'.format(num))
    log.info('end idx_topic')

def get_dict(path):
    with open(path, 'r') as f:
        dd = dict([(key.rstrip(), str(value)) for value, key in enumerate(f)])
        return dd

def _formate_idx_data(in_path, out_path, dd):
    with open(out_path, 'w') as out:
        with open(in_path, 'r') as f:
            for num, line in enumerate(f):
                tmp = [dd[ele] for ele in line.rstrip().split(',')]
                out.write(','.join(tmp) + '\n')
                if num % 100000 == 0:
                    log.info('finished: {}'.format(num))

def format_idx_data():
    log.info('begin format_merge_data')
    word_idx = get_dict(DataPathConfig.get_word_idx_path())
    char_idx = get_dict(DataPathConfig.get_char_idx_path())
    _formate_idx_data(
            DataPathConfig.get_question_train_word_set_path(),
            DataPathConfig.get_question_train_word_idx_path(),
            word_idx)
    _formate_idx_data(
            DataPathConfig.get_question_train_char_set_path(),
            DataPathConfig.get_question_train_char_idx_path(),
            char_idx)
    log.info('end format_merge_data')

def add_element(data, out):
    for ele in data.rstrip('\n').split(','):
        out.add(ele)

def format_question_data():
    q_in = open(RawDataPathConfig.get_question_train_set_path(), 'r')
    qt_in = open(RawDataPathConfig.get_question_topic_train_set_path(), 'r')

    q_w_title_out = open(DataPathConfig.get_question_train_word_title_set_path(), 'w')
    q_w_desc_out = open(DataPathConfig.get_question_train_word_desc_set_path(), 'w')
    q_c_title_out = open(DataPathConfig.get_question_train_character_title_set_path(), 'w')
    q_c_desc_out = open(DataPathConfig.get_question_train_character_desc_set_path(), 'w')
    qt_out = open(DataPathConfig.get_question_topic_train_set_path(), 'w')

    q_w_set_out = open(DataPathConfig.get_question_train_word_set_path(), 'w')
    q_c_set_out = open(DataPathConfig.get_question_train_char_set_path(), 'w')

    q_w_set_topic_split_out = open(DataPathConfig.get_question_train_word_topic_split_set_path(), 'w')
    q_c_set_topic_split_out = open(DataPathConfig.get_question_train_char_topic_split_set_path(), 'w')
    qt_topic_split_out = open(DataPathConfig.get_question_topic_train_topic_split_set_path(), 'w')

    missed_match_count = 0

    t0 = time()
    num = 0

    word_set = set()
    char_set = set()

    while True:
        num += 1
        if num % 300000 == 0:
            log.info('finished: {} in {:.2f}s'.format(num, time() - t0))
        q_l = q_in.readline()
        qt_l = qt_in.readline()
        if q_l == '' and qt_l == '':
            break

        q_items = q_l.rstrip('\n').split('\t')
        qt_items = qt_l.rstrip('\n').split('\t')

        if q_items[0] != qt_items[0]:
            missed_match_count += 1
            continue

        add_element(q_items[2], word_set)
        add_element(q_items[4], word_set)
        add_element(q_items[1], char_set)
        add_element(q_items[3], char_set)

        q_c_title_out.write(q_items[1] + '\n')
        q_c_desc_out.write(q_items[3] + '\n')
        q_w_title_out.write(q_items[2] + '\n')
        q_w_desc_out.write(q_items[4] + '\n')

        qt_out.write(qt_items[1] + '\n')
        q_w_set_out.write(q_items[2] + ',' + q_items[4] + '\n')
        q_c_set_out.write(q_items[1] + ',' + q_items[3] + '\n')

        for topic in qt_items[1].split(','):
            qt_topic_split_out.write(topic + '\n')
            q_w_set_topic_split_out.write(q_items[2] + ',' + q_items[4] + '\n')
            q_c_set_topic_split_out.write(q_items[1] + ',' + q_items[3] + '\n')

    q_in.close()
    qt_in.close()

    q_w_set_out.close()
    q_c_set_out.close()
    q_w_title_out.close()
    q_w_desc_out.close()
    q_c_title_out.close()
    q_c_desc_out.close()
    qt_out.close()

    log.info('begine generate index')
    with open(DataPathConfig.get_word_idx_path(), 'w') as f:
        for ele in word_set:
            f.write(ele + '\n')
    with open(DataPathConfig.get_char_idx_path(), 'w') as f:
        for ele in char_set:
            f.write(ele + '\n')

    log.info('finished all {} in {:.2f}s, missed: {}'.format(num, time() - t0, missed_match_count))

if __name__ == '__main__':
    format_question_data()
    # format_idx_data()
    # format_topic_idx()
