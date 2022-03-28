import torch
import torch.nn.functional as F
from torch import nn

from modules.Utils import SublayerConnection


class DemandPredictionModel(nn.Module):
    def __init__(self,orign_airport_embed,dest_airport_embed,n_layers,output_size, dropout, n_class):
        super(DemandPredictionModel, self).__init__()
        # 起飞机场embedding
        self.orign_airport_embed = orign_airport_embed
        #目的机场embedding
        self.dest_airport_embed = dest_airport_embed

        # 将上述两输出结果进行拼接，输入N层（layer norm + Residual connection 网络），最后经过线性变化输出结果
        self.fc = nn.Sequential(*[SublayerConnection(output_size, dropout) for _ in range(n_layers)],
                                nn.Linear(output_size, n_class))


    def forward(self, h1, origns, dests):
        """
        h1: 对两年销售数据[Batch,730,17]的卷积特征
        origns: 起飞机场
        dests: 目的机场
        """
        xs = torch.cat([h1,self.orign_airport_embed(origns),self.dest_airport_embed(dests)], dim=-1)
        return F.relu(self.fc(xs))


