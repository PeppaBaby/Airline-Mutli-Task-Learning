import numpy as np
import torch
from torch import nn

from data.HistoryStatDataset import HistoryStatDataset


# 旨在针对当日期、构建历史销售矩阵
class HistoryEmbedding(nn.Module):
    # input_shape: [17,730], 前者每天各票价等级的销售量,后者为天数
    def __init__(self, input_shape):
        super(HistoryEmbedding, self).__init__()

        # 得到所有日期的mask矩阵
        masks = self._make_mask(input_shape)
        # 使用register_buffer
        self.register_buffer('masks', masks)

    # 其中ds表示订单的日期, xs订单的730天销售矩阵
    # xs=[ds,xs]
    def forward(self, xs):
        return self.masks[xs[0]] * xs[1]

    # 对于历史销售矩阵[17,730],构建730个矩阵，每个矩阵为[17,730]
    # 其中第i矩阵M，有如此特性: M[:,:i]=1, M[:,i:]=0
    @staticmethod
    def _make_mask(shape):
        # shape:[17,730]
        # 构建矩阵[730,17,730]
        masks = np.ones((shape[1], shape[0], shape[1]))
        # M[:,i+1:]=0
        for i, mask in enumerate(masks[:-1]):  mask[:, i:] = 0

        return torch.Tensor(masks)


# 测试代码是否正常
if __name__ == '__main__':
    dataset = HistoryStatDataset(r'E:/AirCabin/data/count')
    embeddings = HistoryEmbedding((17, 730))
    results = embeddings([torch.LongTensor([2, 3, 4]), dataset[:3]])
    print(results.shape)
