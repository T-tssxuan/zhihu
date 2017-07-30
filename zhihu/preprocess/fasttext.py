import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig, EvalPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('fasttext preprocess')

def generate_eval_data():
    EVAL_SIZE = 217360
    fdesc = open(EvalPathConfig.get_question_eval_word_desc_set_path(), 'r')
    ftitle = open(EvalPathConfig.get_question_eval_word_title_set_path(), 'r')

    feval = open(EvalPathConfig.get_fasttext_eval_word_path(), 'w')

    log.info('generate eval data')
    idx = 0
    for idx in range(EVAL_SIZE):
        desc = ' '.join(fdesc.readline().rstrip().split(','))
        title = ' '.join(ftitle.readline().rstrip().split(','))
        content = desc + ' ' + title
        feval.write(content + '\n')
        idx += 1

        if idx % 100000 == 0:
            log.info('finished {}'.format(idx))

    log.info('finished all {}'.format(idx))
    fdesc.close()
    ftitle.close()
    feval.close()

def generate_train_data():
    TOTAL = 2999967
    TEST_SIZE = 10000

    fdesc = open(DataPathConfig.get_question_train_word_desc_set_path(), 'r')
    ftitle = open(DataPathConfig.get_question_train_word_title_set_path(), 'r')
    ftopic = open(DataPathConfig.get_question_topic_train_set_path(), 'r')

    ftrain = open(DataPathConfig.get_fasttext_train_word_path(), 'w')

    ftest = open(DataPathConfig.get_fasttext_test_word_path(), 'w')
    ftest_topic = open(DataPathConfig.get_fasttext_test_topic_path(), 'w')

    log.info('get train data')
    for i in range(TOTAL - TEST_SIZE):
        desc = ' '.join(fdesc.readline().rstrip().split(','))
        title = ' '.join(ftitle.readline().rstrip().split(','))
        content = desc + ' ' + title

        topics = ftopic.readline().rstrip().split(',')
        label = ' '.join(['__label__' + topic for topic in topics])
        
        ftrain.write(label + ' ' + content + '\n')
        if i % 300000 == 0:
            log.info('finished {:.2f}%'.format(i * 100. / TOTAL))

    log.info('finished {:.2f}%'.format(i * 100. / TOTAL))
    ftrain.close()

    log.info('get test data')
    for i in range(TEST_SIZE):
        desc = ' '.join(fdesc.readline().rstrip().split(','))
        title = ' '.join(ftitle.readline().rstrip().split(','))
        content = desc + ' ' + title
        ftest.write(content + '\n')

        topics = ftopic.readline().rstrip().split(',')
        label = ' '.join(['__label__' + topic for topic in topics])
        ftest_topic.write(label + '\n')

    log.info('finished test data')
    ftopic.close()
    fdesc.close()
    ftitle.close()
    ftest.close()
    ftest_topic.close()

if __name__ == '__main__':
    generate_train_data()
    generate_eval_data()
