import torch.nn.functional as F
from torch import nn

from modules.Utils import SublayerConnection

""" 
[1] 通过HistoryFeature从历史定价数据上学习票价历史时序特征
[2] 通过TicketFeature从当前订单学习特征映射
[3] 拼接上述特征，进行LayerNorm 归一化等操作，残差 connection
[4] 最后同时预测票价等级、票价需求量
"""


class MultiTaskModel(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hist_model, ticket_model, demand_predictor, n_layers, output_size, dropout, n_class):
        super(MultiTaskModel, self).__init__()
        # 历史定价时序特征学习模型
        self.hist_model = hist_model
        # 当前订单的特征学习模型
        self.ticket_model = ticket_model
        # 将上述两输出结果进行拼接，输入N层（layer norm + Residual connection 网络），最后经过线性变化输出结果
        self.fc = nn.Sequential(*[SublayerConnection(output_size, dropout) for _ in range(n_layers)],
                                nn.Linear(output_size, n_class))

        # 利用第1,2.., d-1天的时序特征，预测第d天
        self.demand_predictor = demand_predictor

    def forward(self, hs, ds, xs, origns, dests):
        """
        xs: [Batch,730,17], batch为批大小； 730为两年销售数据； 17为航班总共17个票价等级
        ds: [Batch,] 当前批中每个订单的日期
        tfs:[Batch,nfeature_embedding] 当前批中每个订单的特征(经过embedding之后)
        origns:[Batch,d] 起飞机场embedding
        dests:[Batch,d] 目的机场embedding
        """
        h1, h2 = self.hist_model([ds, hs]), self.ticket_model(xs)
        # 期中demand_predictor旨在多任务调用，以预测第d天的需求量
        return F.log_softmax(self.fc(h1 + h2), dim=-1), self.demand_predictor(h1, origns, dests)
