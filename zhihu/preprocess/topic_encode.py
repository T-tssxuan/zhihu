import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig
from ..utils.tools import Tools
import numpy as np
import queue

log = Tools.get_logger('topic encode')

def get_parent_info():
    log.info('get parent info')
    parent = dict()
    with open(DataPathConfig.get_parents_of_topic_path(), 'r') as f:
        for line in f:
            elements = line.rstrip().split(',')
            parent[elements[0]] = elements[1:]
    print(parent.keys())
    return parent

def get_topic_index():
    log.info('get topic index')
    topic_idx = dict()
    with open(DataPathConfig.get_topic_set_path(), 'r') as f:
        for idx, line in enumerate(f):
            topic_idx[line.rstrip()] = idx
    return topic_idx

no_parent = 0
with_parent = 0
def update_parent(parent, topic_idx, cur_top, vec):
    global no_parent, with_parent
    q = queue.Queue()
    q.put(cur_top)
    while not q.empty():
        ff = q.get()
        vec[topic_idx[ff]] = 1
        if ff in parent:
            with_parent += 1
            for p in parent[ff]:
                q.put(p)
        else:
            no_parent += 1

def topic_encode():
    parent = get_parent_info()
    topic_idx = get_topic_index()
    topic_size = len(topic_idx.keys())

    tfile = open(DataPathConfig.get_topic_with_parent_propagate(), 'w')
    log.info('begin encode')
    with open(DataPathConfig.get_question_topic_train_set_path(), 'r') as f:
        for idx, line in enumerate(f):
            topics = line.rstrip().split(',')
            vec = [0 for i in range(topic_size)]
            for t in topics:
                update_parent(parent, topic_idx, t, vec)
            vec = [str(i) for i in vec]
            tfile.write(','.join(vec) + '\n')
            if idx % 10000 == 0:
                log.info('finshed: {}, no_parent: {}, with_parent: {}'.format(idx, no_parent, with_parent))
            break
    tfile.close()
    log.info('finished encode, no_parent: {}, with_parent: {}'.format(no_parent, with_parent))

if __name__ == '__main__':
    topic_encode()

