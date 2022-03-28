import numpy as np
import os
import torch
from itertools import chain
from torch.utils.data import Dataset

# 导入当前订单数据
class TicketOrderDataset(Dataset):
    # 从directory读取所有航信的历史销售信息
    def __init__(self, dir):
        # 读取目录下所有文件（航线历史销售数据）
        files = os.listdir(dir)
        # 得到所有订票记录，以及每个订单对应的航线
        records = [self._read(os.path.join(dir, f)) for f in files]
        self.airlines = np.array(list(chain(*[len(r) * [f.split('.')[0]] for r, f in zip(records, files)])))

        # 采用_read读取每个文件，然后形成三个array，分别是 订票日期dates、当前订单特征X、预测标签Y
        dates, Xs, Ys = list(zip(*list(chain(*records))))

        # 将日期映射成index
        date2idx = {d: idx for idx, d in enumerate(sorted(set(dates), reverse=False))}

        self.dates = torch.LongTensor([date2idx[d] for d in dates])
        # 将feature映射成index,并得到各维特征的值数
        f2idx = [{f: i for i, f in enumerate(set(fs))} for fs in zip(*Xs)]
        self.nfeatures = [len(fx) for fx in f2idx]
        self.Xs = torch.LongTensor(list(zip(*[[f2idx[i][f] for f in fs] for i, fs in enumerate(zip(*Xs))])))
        # 将不需要做embedding转化tensor
        # self.Xs2 = torch.LongTensor(np.array(Xs2, dtype=np.float32)).cuda(device)
        # 将票价等级标签label转化tensor
        self.Ys = torch.LongTensor(np.array(Ys, dtype=np.float32))

    # override the method __getitem__ in the abstract class Dataset
    def __getitem__(self, index):
        # 返回当前订单的索引
        return [s[index] for s in [self.airlines, self.dates, self.Xs, self.Ys]]

    # override the method __len__ in the abstract class Dataset
    def __len__(self):
        return self.Xs.shape[0]

    # 读取特定的文件，一个文件为一条航线两年的历史记录
    @staticmethod
    def _read(filename):
        # 每一行返回一个三元组(date,x,y)，其中
        with open(filename, 'r') as f:
            # 跳过首行,放弃回车符号:
            tokens = [line.rstrip('\n').split(',') for line in f.readlines()[1:]]
            #丢弃PNRNBR、discount
            [(r.pop(6)) for r in tokens]
            # 返回三元组(date, x(输入),y预测（票价等级))
            return [(r[0], r[2:-2], r[-1]) for r in tokens]


# 测试代码是否正常
if __name__ == '__main__':
    dataset = TicketOrderDataset(r'../airline/PEK_pro')
    print(len(dataset))
    print(dataset[:2])
