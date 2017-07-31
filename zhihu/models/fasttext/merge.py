from ..validate.score import Score
from ...utils.tools import Tools
from ...config.data_path_config import DataPathConfig, EvalPathConfig
import os 

log = Tools.get_logger('merge')

class Merge:
    def __init__(self, data):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.paths = []
        for d in data:
            self.paths.append(cur_path + '/data/' + d)
        self.merged_path = cur_path + '/data/merged.txt'

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

            mf.write(','.join(list(count)) + '\n')
        for f in files:
            f.close()
        mf.close()
        log.info('finieshed top')

    def generate_merge_file(self, min_count=0):
        if min_count == 0:
            min_count = len(self.paths)
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
    m = Merge(['test_char_300_256_10.txt', 'test_word_200_128_10.txt', 'test_word_150_128_10.txt'])
    # m.generate_merge_file(min_count=0)
    m.generate_top_file()
    score = m.score()
    log.info(score)

