import os 
import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider, PropagatedTopicProvider
from ...config.data_path_config import DataPathConfig, EvalPathConfig
from ...utils.tools import Tools
from ..validate.score import Score

dir_path = os.path.dirname(os.path.realpath(__file__))

log = Tools.get_logger('FastText')

class FastText:
    def __init__(self, category='word'):
        model_prefix = dir_path + '/model/model'
        cmd_path = dir_path + '/fastText/fasttext'
        self.train_cmd_fmt = cmd_path + ' supervised -input {} -output ' + model_prefix

        setting = ' -epoch 50 -thread 20 -dim 256 '
        self.train_cmd_fmt += setting

        model_path = dir_path + '/model/model.bin'
        self.output_file = dir_path + '/data/rst.txt'
        self.predict_cmd_fmt = cmd_path + ' predict ' + model_path + ' {} {} > {}'

        self.train_data = DataPathConfig.get_fasttext_train_word_path()
        self.test_data = DataPathConfig.get_fasttext_test_word_path()
        self.eval_data = EvalPathConfig.get_fasttext_eval_word_path()
        if category == 'char':
            pass

    def train(self):
        cmd = self.train_cmd_fmt.format(self.train_data)
        log.info(cmd)
        os.system(cmd)

    def test(self):
        cmd = self.predict_cmd_fmt.format(self.test_data, 5, self.output_file)
        log.info(cmd)
        os.system(cmd)


    def score():
        pass

if __name__ == '__main__':
    ft = FastText();
    ft.train()
    ft.test()
