import os 
import tensorflow as tf
import numpy as np
from ..data.data_provider import DataProvider, TopicProvider, PropagatedTopicProvider
from ...config.data_path_config import DataPathConfig
from ...utils.tools import Tools
from ..validate.score import Score

dir_path = os.path.dirname(os.path.realpath(__file__))

class FastText:
    def __init__(self):
        self.fasttext_model_path = dir_path + '/model/model'
        self.fasttext_cmd_fmt = dir_path + '/fastText/fasttext skipgram -input {} -output {}'

    def train(self):
        cmd = self.fasttext_cmd_fmt.format(
                DataPathConfig.get_question_fasttext_doc_path(),
                self.fasttext_model_path)
        os.system(cmd)

    def score():
