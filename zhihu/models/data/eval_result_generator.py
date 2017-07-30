from ...config.data_path_config import EvalPathConfig
import gensim
import numpy as np
from ...utils.tools import Tools

log = Tools.get_logger('EvalResultGenerator')

class EvalResultGenerator:
    def __init__(self):
        pass

    def generate(self, path):
        log.info('begin')

        fpre = open(path, 'r')
        ftid = open(EvalPathConfig.get_eval_tid_path(), 'r')
        fresult = open(EvalPathConfig.get_eval_result_path(), 'w')
        while True:
            pre = fpre.readline().rstrip()
            tid = ftid.readline().rstrip()
            if tid == '':
                break
            fresult.write(tid + ',' + pre + '\n')

        fpre.close()
        ftid.close()
        fresult.close()

        log.info('finished')
