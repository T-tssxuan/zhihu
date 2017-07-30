import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig, EvalPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('fasttext preprocess')

def generate_word_eval():
    log.info('generate_word_eval')
    generate_eval_data(
            EvalPathConfig.get_question_eval_word_desc_set_path(),
            EvalPathConfig.get_question_eval_word_title_set_path(),
            EvalPathConfig.get_fasttext_eval_word_path()
            )
    log.info('finished generate_word_eval')

def generate_char_eval():
    log.info('generate_char_eval')
    generate_eval_data(
            EvalPathConfig.get_question_eval_character_desc_set_path(),
            EvalPathConfig.get_question_eval_character_title_set_path(),
            EvalPathConfig.get_fasttext_eval_character_path()
            )
    log.info('finished generate_char_eval')


def generate_eval_data(desc_path, title_path, eval_path):
    EVAL_SIZE = 217360
    fdesc = open(desc_path, 'r')
    ftitle = open(title_path, 'r')

    feval = open(eval_path, 'w')

    log.info('generate eval data')
    idx = 0
    for idx in range(EVAL_SIZE):
        desc = ' '.join(fdesc.readline().rstrip().split(','))
        title = ' '.join(ftitle.readline().rstrip().split(','))
        content = desc + ' ' + title
        feval.write(content + '\n')
        idx += 1

        if idx % 30000 == 0:
            log.info('finished {}'.format(idx))

    log.info('finished all {}'.format(idx))
    fdesc.close()
    ftitle.close()
    feval.close()

def generate_word_train():
    log.info('generate_word_train')
    generate_train_data(
            DataPathConfig.get_question_train_word_desc_set_path(),
            DataPathConfig.get_question_train_word_title_set_path(),
            DataPathConfig.get_question_topic_train_set_path(),
            DataPathConfig.get_fasttext_train_word_path(),
            DataPathConfig.get_fasttext_test_word_path(),
            DataPathConfig.get_fasttext_test_topic_path()
            )
    log.info('finshed generate_word_train')

def generate_char_train():
    log.info('generate_char_train')
    generate_train_data(
            DataPathConfig.get_question_train_character_desc_set_path(),
            DataPathConfig.get_question_train_character_title_set_path(),
            DataPathConfig.get_question_topic_train_set_path(),
            DataPathConfig.get_fasttext_train_char_path(),
            DataPathConfig.get_fasttext_test_char_path(),
            DataPathConfig.get_fasttext_test_topic_path()
            )
    log.info('finished generate_char_train')

def generate_train_data(desc_path, title_path, topic_path, train_path,
        test_path, test_topic_path):
    TOTAL = 2999967
    TEST_SIZE = 10000

    fdesc = open(desc_path, 'r')
    ftitle = open(title_path, 'r')
    ftopic = open(topic_path, 'r')

    ftrain = open(train_path, 'w')

    ftest = open(test_path, 'w')
    ftest_topic = open(test_topic_path, 'w')

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

        ftest_topic.write(ftopic.readline())

    log.info('finished test data')
    ftopic.close()
    fdesc.close()
    ftitle.close()
    ftest.close()
    ftest_topic.close()

if __name__ == '__main__':
    generate_char_train()
    generate_word_train()
    generate_char_eval()
    generate_word_eval()
