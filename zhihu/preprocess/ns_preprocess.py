import os
from ..config.data_path_config import DataPathConfig, RawDataPathConfig, EvalPathConfig
from ..utils.tools import Tools

log = Tools.get_logger('negative sampling')

def ns_data_generate(content_path, topic_path, output_path):
    cf = open(content_path, 'r')
    tf = open(topic_path, 'r')
    of = open(output_path, 'w')

    for lc, lt in zip(cf, tf):
        topics = lt.rstrip('

