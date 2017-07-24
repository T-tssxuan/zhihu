# 2017/7/10
"""
Generate a dictionary for all topic direct parent topic and the directly child topics
"""
import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('preprocess')

def get_topic_relation():
    log.info('begin')

    children = dict()
    tfile = open(DataPathConfig.get_topic_desc_path(), 'w')
    cfile = open(DataPathConfig.get_children_of_topic_path(), 'w')
    pfile = open(DataPathConfig.get_parents_of_topic_path(), 'w')

    topics = set()

    with open(RawDataPathConfig.get_topic_info_path()) as f:
        for line in f:
            elements = line.rstrip('\n').split('\t')
            elements = [ele.rstrip(' ') for ele in elements]

            tmp = [elements[0]] + elements[2:]
            tfile.write('\t'.join(tmp) + '\n')

            tmp = [elements[0]] + elements[1].split(',')
            pfile.write(','.join(tmp) + '\n')

            if elements[1] != '':
                for ele in elements[1].split(','):
                    if ele not in children:
                        children[ele] = set()
                    children[ele].add(elements[0])
            topics.add(elements[0])

        for key in children:
            cfile.write(','.join([key] + list(children[key])) + '\n')

    tfile.close()
    pfile.close()
    cfile.close()

    with open(DataPathConfig.get_topic_set_path(), 'w') as f:
        for topic in topics:
            f.write(topic + '\n')

    log.info('finished')

if __name__ == '__main__':
    print(os.getcwd())
    get_topic_relation()
