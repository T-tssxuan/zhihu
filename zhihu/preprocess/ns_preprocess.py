import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig, EvalPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('negative sampling')

def ns_data_generate(content_path, topic_path, output_path):
    log.info(output_path)
    cf = open(content_path, 'r')
    tf = open(topic_path, 'r')
    of = open(output_path, 'w')

    num = 0
    row = 0
    for lc, lt in zip(cf, tf):
        row += 1
        for topic in lt.rstrip().split(','):
            of.write(topic + ',' + lc)
            num += 1
        if row % 300000 == 0:
            log.info('finished row: {}, total: {}'.format(row, num))
    cf.close()
    of.close()
    log.info('end')

if __name__ == '__main__':
    ns_data_generate(
            DataPathConfig.get_question_train_word_idx_path(),
            DataPathConfig.get_topic_idx_path(),
            DataPathConfig.get_ns_train_word_path())
    ns_data_generate(
            DataPathConfig.get_question_train_char_idx_path(),
            DataPathConfig.get_topic_idx_path(),
            DataPathConfig.get_ns_train_char_path())

            

