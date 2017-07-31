import os 
import tensorflow as tf
import numpy as np
import tempfile
from ..data.data_provider import DataProvider, TopicProvider, PropagatedTopicProvider
from ...config.data_path_config import DataPathConfig, EvalPathConfig
from ...utils.tools import Tools
from ..validate.score import Score
from ..data.eval_result_generator import EvalResultGenerator

log = Tools.get_logger('FastText')

class FastText:
    def __init__(self, dir_path, category='word', epoch=100, thread=30, dim=128,
            lr=0.05, update_rate=100, model='model'):
        self.suffix = '{}_{}_{}_{:.2f}_{}'.format(category, epoch, dim, lr, update_rate)
        self.dir_path = dir_path
        model_prefix = self.dir_path + '/model/' + model
        cmd_path = os.path.dirname(os.path.realpath(__file__)) + '/fastText/fasttext'
        self.train_cmd_fmt = cmd_path + ' supervised -input {} -output ' + model_prefix

        setting = ' -epoch {} -thread {} -dim {} -lr {} -lrUpdateRate {} '.format(epoch, thread, dim, lr, update_rate)
        self.train_cmd_fmt += setting

        model_path = self.dir_path + '/model/' + model + '.bin'
        self.predict_cmd_fmt = cmd_path + ' predict ' + model_path + ' {} {} > {}'

        self.test_output_file = self.dir_path + '/data/test.txt'
        self.eval_output_file = self.dir_path + '/data/eval.txt'

        self.train_data = DataPathConfig.get_fasttext_train_word_path()
        self.test_data = DataPathConfig.get_fasttext_test_word_path()
        self.eval_data = EvalPathConfig.get_fasttext_eval_word_path()
        if category == 'char':
            self.train_data = DataPathConfig.get_fasttext_train_char_path()
            self.test_data = DataPathConfig.get_fasttext_test_char_path()
            self.eval_data = EvalPathConfig.get_fasttext_eval_character_path()

    def train(self):
        cmd = self.train_cmd_fmt.format(self.train_data)
        log.info(cmd)
        os.system(cmd)

    def _output_format(self, input_path):
        log.info('begin format data')
        with tempfile.TemporaryFile() as fp:
            tmp = len('__lable__')
            with open(input_path, 'r') as f:
                for line in f:
                    elements = [ele[tmp:] for ele in line.rstrip().split(' ')]
                    ss = ','.join(elements) + '\n'
                    fp.write(ss.encode())
            fp.seek(0)
            with open(input_path, 'w') as f:
                for line in fp:
                    f.write(line.decode())
        log.info('finished format data')

    def test(self, topk=5):
        log.info('generate test result')
        cmd = self.predict_cmd_fmt.format(self.test_data, topk, self.test_output_file)
        log.info(cmd)
        os.system(cmd)
        log.info('finished generate test result')

        self._output_format(self.test_output_file)

        s = self._test_score()
        log.info('test set eval score: {}'.format(self._test_score()))

    def _test_score(self):
        pre = []
        with open(self.test_output_file, 'r') as f:
            for line in f:
                pre.append(line.rstrip().split(','))

        src = []
        with open(DataPathConfig.get_fasttext_test_topic_path()) as f:
            for line in f:
                src.append(line.rstrip().split(','))

        score = Score(topk=5)
        return score.vanilla_score(pre, src)

    def eval(self, topk=5):
        log.info('begin generate eval result')
        cmd = self.predict_cmd_fmt.format(self.eval_data, topk, self.eval_output_file)
        log.info(cmd)
        os.system(cmd)
        self._output_format(self.eval_output_file)

        eg = EvalResultGenerator()
        eg.generate(self.eval_output_file, self.suffix)

        log.info('finished generate eval result')
        
if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.realpath(__file__))
    ft = FastText('word', cur_path);
    ft.train()
    ft.test()
    ft.eval()
