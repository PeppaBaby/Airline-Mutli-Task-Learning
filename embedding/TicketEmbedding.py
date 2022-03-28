import math
import numpy as np
import torch
from torch import nn


# 分别对Hometown,birthday,gender,airline,is_holiday,orignial, destination 进行embedding

class TicketEmbedding(nn.Module):
    # features则表示每个属性有多少取值, embeded_size映射后的维度
    def __init__(self, features, embeded_size):
        super(TicketEmbedding, self).__init__()
        # 当embeded_size为一个整数时，将其变成list
        if not isinstance(embeded_size, list):
            embeded_size = [embeded_size] * len(features)
        self.embedings = nn.ModuleList([nn.Embedding(f, d) for f, d in zip(features, embeded_size)])
        # 总的feature大小
        self.d_model = np.sum([f * d for f, d in zip(features, embeded_size)])

    # 将每个field特征进行拼接
    def forward(self, xs):
        # xs: 一行为一个样本，为了加快embedding index，对其进行转置
        # embed_x1 = torch.cat([self.embedings[i](f) for i, f in enumerate(xs1.T)], dim=-1) / math.sqrt(self.d_model)
        # x = torch.cat([embed_x1,xs2],dim=0)
        # return x
        return torch.cat([self.embedings[i](f) for i, f in enumerate(xs.T)], dim=-1) * math.sqrt(self.d_model)


# 测试代码是否正常
if __name__ == '__main__':
    embeddings = TicketEmbedding([4, 5, 6], 4)
    index = torch.LongTensor([[1, 2, 3], [1, 3, 5], [1, 4, 5]])
    print(embeddings(index))
