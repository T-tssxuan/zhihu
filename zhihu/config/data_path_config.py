import os
from ..utils.tools import Tools

class RawDataPathConfig:
    def get_topic_info_path():
        return Tools.get_raw_data_direcory() + 'topic_info.txt'
    def get_question_train_set_path():
        return Tools.get_raw_data_direcory() + 'question_train_set.txt'
    def get_question_topic_train_set_path():
        return Tools.get_raw_data_direcory() + 'question_topic_train_set.txt'
    def get_question_eval_set_path():
        return Tools.get_raw_data_direcory() + 'question_eval_set.txt'

class DataPathConfig:
    def get_topic_desc_path():
        return Tools.get_data_directory() + 'topic_desc.tsv'
    def get_topic_desc_idx_path():
        return Tools.get_data_directory() + 'topic_desc_idx.csv'
    def get_children_of_topic_path():
        return Tools.get_data_directory() + 'children_of_topic.csv'
    def get_parents_of_topic_path():
        return Tools.get_data_directory() + 'parents_of_topic.csv'
    def get_topic_set_path():
        return Tools.get_data_directory() + 'topic_set.txt'
    def get_topic_idx_path():
        return Tools.get_data_directory() + 'topic_idx.txt'

    def get_word_idx_path():
        return Tools.get_data_directory() + 'word_idx.txt'
    def get_char_idx_path():
        return Tools.get_data_directory() + 'char_idx.txt'
    def get_question_train_word_idx_path():
        return Tools.get_data_directory() + 'question_train_word_idx.txt'
    def get_question_train_char_idx_path():
        return Tools.get_data_directory() + 'question_train_char_idx.txt'

    def get_question_train_word_set_path():
        return Tools.get_data_directory() + 'question_train_word_set.txt'
    def get_question_train_char_set_path():
        return Tools.get_data_directory() + 'question_train_char_set.txt'

    def get_question_train_word_topic_split_set_path():
        return Tools.get_data_directory() + 'question_train_word_topic_split_set.csv'
    def get_question_train_char_topic_split_set_path():
        return Tools.get_data_directory() + 'question_train_char_topic_split_set.csv'
    def get_question_topic_train_topic_split_set_path():
        return Tools.get_data_directory() + 'question_topic_train_topic_split_set.csv'

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

    def get_fasttext_train_word_path():
        return Tools.get_data_directory() + 'fasttext_train_word.txt'
    def get_fasttext_test_word_path():
        return Tools.get_data_directory() + 'fasttext_test_word.txt'
    def get_fasttext_train_char_path():
        return Tools.get_data_directory() + 'fasttext_train_char.txt'
    def get_fasttext_test_char_path():
        return Tools.get_data_directory() + 'fasttext_test_char.txt'
    def get_fasttext_test_topic_path():
        return Tools.get_data_directory() + 'fasttext_test_topic.txt'

    def get_ns_train_word_path():
        return Tools.get_data_directory() + 'ns_train_word.txt'
    def get_ns_train_char_path():
        return Tools.get_data_directory() + 'ns_train_char.txt'



class EvalPathConfig:
    def get_question_eval_word_title_set_path():
        return Tools.get_data_directory() + 'question_eval_word_title_set.csv'
    def get_question_eval_word_desc_set_path():
        return Tools.get_data_directory() + 'question_eval_word_desc_set.csv'
    def get_fasttext_eval_word_path():
        return Tools.get_data_directory() + 'fasttext_eval_word.txt'

    def get_question_eval_character_title_set_path():
        return Tools.get_data_directory() + 'question_eval_character_title_set.csv'
    def get_question_eval_character_desc_set_path():
        return Tools.get_data_directory() + 'question_eval_character_desc_set.csv'
    def get_fasttext_eval_character_path():
        return Tools.get_data_directory() + 'fasttext_eval_character.txt'

    def get_eval_tid_path():
        return Tools.get_data_directory() + 'eval_tid.txt'

    def get_eval_result_path(suffix=''):
        if not os.path.exists(Tools.get_data_directory() + '/result/'):
            os.mkdir(Tools.get_data_directory() + '/result/')
        return Tools.get_data_directory() + '/result/eval_result_' + suffix + '.csv'

if __name__ == '__main__':
    print(DataPathConfig.get_raw_topic_info_path())
    print(DataPathConfig.get_topic_embedding_path())
