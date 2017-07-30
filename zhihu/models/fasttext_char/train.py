from ..fasttext.fasttext import FastText
from ...utils.tools import Tools
import os, sys

log = Tools.get_logger('fasttext char')

class FastTextChar:
    def __init__(self, epoch=100, thread=30, dim=128):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(cur_path + '/model'):
            os.mkdir(cur_path + '/model')
        if not os.path.exists(cur_path + '/data'):
            os.mkdir(cur_path + '/data')
        model = 'model_{}_{}'.format(epoch, dim)
        self.obj = FastText(cur_path, 'char', epoch, thread, dim, model)

    def train(self):
        log.info('begin train')
        self.obj.train()
        log.info('end train')

    def test(self):
        log.info('begin test')
        self.obj.test()
        log.info('end test')

    def eval(self):
        log.info('begin eval')
        self.obj.eval()
        log.info('end eval')

if __name__ == '__main__':
    ftc = FastTextChar(epoch=200, thread=30)
    ftc.train()
    ftc.test()
    ftc.eval()
