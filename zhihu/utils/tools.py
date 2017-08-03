import os 
import logging
import sys

class Tools:
    log_file = ''
    def get_tf_summary_path():
        path = Tools.get_project_directory() + '/tfboard/'
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_project_directory():
        return os.getcwd() + '/zhihu/'

    def get_raw_data_direcory():
        return os.getcwd() + '/zhihu/data/raw/'

    def get_data_directory():
        return os.getcwd() + '/zhihu/data/'

    def set_log_file(file_name):
        Tools.log_file = Tools.get_project_directory() + '/log/' + file_name 

    def get_logger(name='defualt', level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        fmt = '%(asctime)s %(levelname)-8s [' + name + '] %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        if Tools.log_file != '':
            handler = logging.FileHandler('log.txt', mode='a')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
        return logger

if __name__ == '__main__':
    print(Tools.get_project_directory())
    print(Tools.get_raw_data_direcory())
    print(Tools.get_data_directory())
    log = Tools.get_logger()
    log.info('test')
