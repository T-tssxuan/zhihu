import os
from ..utils.tools import Tools

class RawDataPathConfig:
    def get_topic_info_path():
        return Tools.get_raw_data_direcory() + 'topic_info.txt'
    def get_question_train_set_path():
        return Tools.get_raw_data_direcory() + 'question_train_set.txt'
    def get_question_topic_train_set_path():
        return Tools.get_raw_data_direcory() + 'question_topic_train_set.txt'

class DataPathConfig:
    def get_topic_desc_path():
        return Tools.get_data_directory() + 'topic_desc.tsv'
    def get_children_of_topic_path():
        return Tools.get_data_directory() + 'children_of_topic.csv'
    def get_parents_of_topic_path():
        return Tools.get_data_directory() + 'parents_of_topic.csv'
    def get_topic_set_path():
        return Tools.get_data_directory() + 'topic_set.txt'

    def get_question_train_word_title_set_path():
        return Tools.get_data_directory() + 'question_train_word_title_set.csv'
    def get_question_train_word_desc_set_path():
        return Tools.get_data_directory() + 'question_train_word_desc_set.csv'
    def get_question_train_character_title_set_path():
        return Tools.get_data_directory() + 'question_train_character_title_set.csv'
    def get_question_train_character_desc_set_path():
        return Tools.get_data_directory() + 'question_train_character_desc_set.csv'

    def get_question_topic_train_set_path():
        return Tools.get_data_directory() + 'question_topic_train_set.csv'

    def get_word_embedding_path():
        return Tools.get_data_directory() + 'word_embedding.txt'
    def get_char_embedding_path():
        return Tools.get_data_directory() + 'char_embedding.txt'
    def get_topic_to_topic_matrix_path():
        return Tools.get_data_directory() + 'topic_to_topic.mtx'

    def get_topic_with_parent_propagate():
        return Tools.get_data_directory() + 'topic_with_parent_propagate.csv'

    def get_question_tfidf_vec_path():
        return Tools.get_data_directory() + 'tfidf.vec'

    def get_question_fasttext_doc_path():
        return Tools.get_data_directory() + 'fasttext.txt'

if __name__ == '__main__':
    print(DataPathConfig.get_raw_topic_info_path())
    print(DataPathConfig.get_topic_embedding_path())
