import os
from ..utils.tools import Tools

class DataPathConfig:
    def get_raw_topic_info_path():
        return Tools.get_raw_data_direcory() + 'topic_info.txt'

    def get_topic_desc_path():
        return Tools.get_data_directory() + 'topic_desc.csv'
    def get_children_of_topic_path():
        return Tools.get_data_directory() + 'children_of_topic.csv'
    def get_parents_of_topic_path():
        return Tools.get_data_directory() + 'parents_of_topic.csv'

if __name__ == '__main__':
    print(DataPathConfig.get_raw_topic_info_path())
    print(DataPathConfig.get_topic_embedding_path())
