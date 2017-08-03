import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig, EvalPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('negative sampling')

def ns_data_generate(content_path, topic_path, output_path):
    log.info(out_path)
    cf = open(content_path, 'r')
    tf = open(topic_path, 'r')
    of = open(output_path, 'w')

    for lc, lt in zip(cf, tf):
        for topic in lt.rstrip().split(','):
            of.write(topic + ',' + lc)
    cf.close()
    of.close()
    log.info('end')

if __name__ == '__main__':
    ns_data_generate(
            DataPathConfig.get_question_train_word_idx_path(),
            DataPathConfig.get_topic_desc_idx_path(),
            DataPathConfig.get_ns_train_word_path())
    ns_data_generate(
            DataPathConfig.get_question_train_char_idx_path(),
            DataPathConfig.get_topic_desc_idx_path(),
            DataPathConfig.get_ns_train_char_path())

            

