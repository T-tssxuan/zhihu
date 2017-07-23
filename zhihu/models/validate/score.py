import numpy as np
import math

class Score:
    def __init__(self):
        pass

    def _eval(self, predict_label_and_marked_label_list):
        """
        :param predict_label_and_marked_label_list: 一个元组列表。例如
        [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
          ([3, 2, 1, 4, 7], [5, 7, 3])
         ]
        需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

        marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
        [0，0，0，1，1]   (4，5命中)
        [1，0，0，0，1]   (3，7命中)

        """
        right_label_num = 0  #总命中标签数量
        right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
        sample_num = 0   #总问题数量
        all_marked_label_num = 0    #总标签数量
        for predict_labels, marked_labels in predict_label_and_marked_label_list:
            sample_num += 1
            marked_label_set = set(marked_labels)
            all_marked_label_num += len(marked_label_set)
            for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
                if label in marked_label_set:     #命中
                    right_label_num += 1
                    right_label_at_pos_num[pos] += 1

        precision = 0.0
        for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
            precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
        recall = float(right_label_num) / all_marked_label_num

        return (precision * recall) / max((precision + recall), 0.000001)

    def score(self, pre, src):
        pre = np.array(pre)
        pre = pre.argsort()[:, -5:][:, ::-1].tolist()
        src = np.array(src)
        src = [list(np.where(ele > 0)[0]) for ele in src]
        merged = [(list(set(a)), list(set(b))) for a, b in zip(pre, src)]
        return self._eval(merged)

if __name__ == '__main__':
    pred = np.random.randint(0, 1000, size=(100, 5))
    src = np.random.randint(0, 2, size=(100, 10))
    v = Score()
    score = v.score(pred, src)
    print(score)
