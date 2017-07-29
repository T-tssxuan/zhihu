import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('fast text prepare')
TOTAL = 2999967
def generate_doc():
    fdesc = open(DataPathConfig.get_question_train_word_desc_set_path(), 'r')
    ftitle = open(DataPathConfig.get_question_train_word_title_set_path(), 'r')
    ftopic = open(DataPathConfig.get_question_topic_train_set_path(), 'r')

    output = open(DataPathConfig.get_question_fasttext_doc_path(), 'w')

    for i in range(TOTAL):
        desc = fdesc.readline()
        title = ftitle.readline()
        content = desc + ' ' + title

        topics = ftopic.readline().rstrip().split(',')
        label = ' '.join(['__label__' + topic for topic in topics])
        
        output.write(label + ' ' + content + '\n')
        if i % 300000 == 0:
            log.info('finished {:.2f}'.format(i / TOTAL * 100))

    ftopic.close()
    fdesc.close()
    ftitle.close()
    output.close()

if __name__ == '__main__':
    generate_doc()
