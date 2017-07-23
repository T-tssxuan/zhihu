import numpy as np
from ..utils.tools import Tools
from ..config.data_path_config import DataPathConfig, RawDataPathConfig

def generate_topic_to_topic_matrix():
    log = Tools.get_logger('ttt matrix')
    log.info('begin generate')
    tdict = dict()
    with open(DataPathConfig.get_topic_set_path(), 'r') as f:
        for idx, line in enumerate(f):
            tdict[line.rstrip()] = idx
    mtx = np.zeros((len(tdict), len(tdict)))
    with open(DataPathConfig.get_question_topic_train_set_path(), 'r') as f:
        for line in f:
            eles = line.rstrip().split(',')
            for e1 in eles:
                for e2 in eles:
                    if e1 != e2:
                        mtx[tdict[e1], tdict[e2]] += 1
    np.save(DataPathConfig.get_topic_to_topic_matrix_path(), mtx)
    log.info('finished generate')

if __name__ == '__main__':
    generate_topic_to_topic_matrix()

