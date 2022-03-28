import numpy as np
import os
import torch
from torch.utils.data import Dataset

# 导入历史数据
class HistoryStatDataset(Dataset):
    # 从dir读取所有航信的历史销售信息
    def __init__(self, dir):
        # 读取目录下所有文件（航线历史销售数据）
        files = os.listdir(dir)
        # 采用_read读取每个文件，然后将形成一个三维矩阵  航线数*730天(2年)*17个票价等级
        data = torch.tensor([self._read(os.path.join(dir, f)) for f in files])
        # 为了后续Conv1d方便，将矩阵进行转置
        self.data = data.permute(0, 2, 1)
        # 航线名与索引对应
        self.air2idx = {file.split('.')[0]: index for (index, file) in enumerate(files)}
        # 起飞机场
        origin_airports = [file.split('.')[0][:3] for file in files]
        # 起飞机场映射
        o2idx = {v: k for k, v in enumerate(set(origin_airports))}
        # 起飞机场ID
        self.origin_airports = torch.LongTensor([o2idx[v] for v in origin_airports])

        # 目的机场
        dest_airports = [file.split('.')[0][5:] for file in files]
        # 目的机场映射
        d2idx = {v: k for k, v in enumerate(set(dest_airports))}
        # 目的机场ID
        self.dest_airports = torch.LongTensor([d2idx[v] for v in dest_airports])

        self.origin_vocab,self.dest_vocab = len(o2idx),len(d2idx)

    # override the method __getitem__ in the abstract class Dataset
    def __getitem__(self, index):
        # 先将航线名称映射成索引,然后再根据索引查找航线历史数据信息
        return [s[index] for s in [self.data, self.origin_airports, self.dest_airports]]

    # override the method __len__ in the abstract class Dataset
    def __len__(self):
        return len(self.air2idx)

    # 读取特定的文件，一个文件为一条航线两年的历史记录
    @staticmethod
    def _read(filename):
        with open(filename, 'r') as f:
            # 跳过首行,放弃回车符号、丢弃日期
            return [list(map(float, line.rstrip('\n').split(',')[1:])) for line in f.readlines()[1:]]


# 测试代码是否正常
if __name__ == '__main__':
    dataset = HistoryStatDataset(r'../airline/count')
    print((dataset[:3]))
    print(dataset.air2idx)
