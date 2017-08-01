from ..fasttext.fasttext import FastText
from ...utils.tools import Tools
import os, sys

log = Tools.get_logger('fasttext word')

class FastTextWord:
    def __init__(self, epoch=100, thread=30, dim=128, lr=0.5, update_rate=100, topk=5):
        self.topk = topk
        cur_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(cur_path + '/model'):
            os.mkdir(cur_path + '/model')
        if not os.path.exists(cur_path + '/data'):
            os.mkdir(cur_path + '/data')
        model = 'model_{}_{}_{:.2f}_{}'.format(epoch, dim, lr, update_rate)
        self.obj = FastText(cur_path, 'word', epoch, thread, dim, lr, update_rate, model)

    def train(self):
        log.info('begin train')
        self.obj.train()
        log.info('end train')

    def test(self):
        log.info('begin test')
        self.obj.test(topk=self.topk)
        log.info('end test')

    def eval(self):
        log.info('begin eval')
        self.obj.eval(topk=self.topk)
        log.info('end eval')

if __name__ == '__main__':
    ftc = FastTextWord(epoch=5, thread=30, dim=128, lr=0.5, update_rate=100, topk=5)
    if len(sys.argv) > 2:
        epoch = int(sys.argv[1])
        thread = int(sys.argv[2])
        dim = int(sys.argv[3])
        lr = float(sys.argv[4])
        update_rate = int(sys.argv[5])
        ftc = FastTextWord(epoch=epoch,
                           thread=thread,
                           dim=dim,
                           lr=lr,
                           update_rate=update_rate)
    if sys.argv[6] == 'train' or sys.argv[6] == 'all':
        ftc.train()
    if sys.argv[6] == 'test' or sys.argv[6] == 'all':
        ftc.test()
    if sys.argv[6] == 'eval' or sys.argv[6] == 'all':
        ftc.eval()
