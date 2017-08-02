from ..validate.score import Score
from ...utils.tools import Tools
from ...config.data_path_config import DataPathConfig, EvalPathConfig
import sys
import os 
import glob

log = Tools.get_logger('merge')

class Merge:
    def __init__(self, path_prefix):
        log.info('path_prefix: {}'.format(path_prefix))
        self.paths = glob.glob(path_prefix)
        log.info(self.paths)
        self.merged_path = cur_path + '/data/merged.csv'
        log.info(self.merged_path)

    def generate_top_file(self):
        files = []
        for path in self.paths:
            files.append(open(path, 'r'))

        mf = open(self.merged_path, 'w')
        flag = True
        log.info('begin top')
        while flag:
            count = set()
            for f in files:
                line = f.readline()
                if line == '':
                    flag = False
                    break
                topics = line.rstrip().split(',')
                count.add(topics[0])

            rst = rst[:5]
            if len(rst) < 5:
                rst += ['-1'] * (5 - len(rst))
            mf.write(','.join(list(count)[:5]) + '\n')
        for f in files:
            f.close()
        mf.close()
        log.info('finieshed top')

    def generate_merge_file(self, min_count=0):
        log.info('test merge')
        if min_count == 0:
            min_count = len(self.paths)
        log.info('min_count: {}'.format(min_count))
        files = []
        for path in self.paths:
            files.append(open(path, 'r'))

        mf = open(self.merged_path, 'w')
        flag = True
        log.info('begin merge')
        while flag:
            count = dict()
            for f in files:
                line = f.readline()
                if line == '':
                    flag = False
                    break
                topics = line.rstrip().split(',')
                for t in topics:
                    if t not in count:
                        count[t] = 0
                    count[t] += 1
            rst = []
            for key in count:
                if count[key] >= min_count:
                    rst.append(key)
            rst = rst[:5]
            if len(rst) < 5:
                rst += ['-1'] * (5 - len(rst))
            mf.write(','.join(rst) + '\n')
        for f in files:
            f.close()
        mf.close()
        log.info('finieshed merge')

    def generate_eval_file(self, min_count=0):
        log.info('eval merge')
        if min_count == 0:
            min_count = len(self.paths)
        log.info('min_count: {}'.format(min_count))
        files = []
        for path in self.paths:
            files.append(open(path, 'r'))

        mf = open(self.merged_path, 'w')
        flag = True
        log.info('begin merge')
        while flag:
            count = dict()
            tid = ''
            for f in files:
                line = f.readline()
                if line == '':
                    flag = False
                    break
                topics = line.rstrip().split(',')
                tid = topics[0]
                for t in topics[1:]:
                    if t not in count:
                        count[t] = 0
                    count[t] += 1
            rst = [tid]
            for key in count:
                if count[key] >= min_count:
                    rst.append(key)
            rst = rst[:6]
            if len(rst) < 6:
                rst += ['-1'] * (6 - len(rst))
            mf.write(','.join(rst) + '\n')
        for f in files:
            f.close()
        mf.close()
        log.info('finieshed merge')

    def score(self):
        pre = []
        with open(self.merged_path, 'r') as f:
            for line in f:
                pre.append(line.rstrip().split(','))

        src = []
        with open(DataPathConfig.get_fasttext_test_topic_path()) as f:
            for line in f:
                src.append(line.rstrip().split(','))

        score = Score(topk=5)
        return score.vanilla_score(pre, src)

if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.realpath(__file__))
    path_prefix = cur_path + '/data/test*'
    min_count = 0

    if len(sys.argv) >= 2:
        min_count = int(sys.argv[1])

    if len(sys.argv) >= 3:
        m = Merge(sys.argv[2])
    else:
        m = Merge(path_prefix)

    if min_count == -1:
        m.generate_top_file()
    else:
        if len(sys.argv) >= 3:
            m.generate_eval_file(min_count=min_count)
        else:
            m.generate_merge_file(min_count=min_count)
    if len(sys.argv) < 3:
        score = m.score()
        log.info(score)
    log.info('finished')

