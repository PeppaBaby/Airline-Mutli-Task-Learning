from torch import nn

from .Utils import PositionwiseFeedForward


# 从当前订单中学习用于”票价等级预测“的类别标签
class TicketFeature(nn.Module):
    def __init__(self, input_size, hid_size, dropout=0.1):
        # 经过3个线性层变化 :
        # (1)第1层 input_size ->hid_size    x1 = max(w1*x+b1,0)
        # (2)第2层: hid_size ->size  w3*( max(w2*x1+b2,0)
        super(TicketFeature, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, hid_size),
                                nn.ReLU(),
                                PositionwiseFeedForward(hid_size, hid_size, dropout))

    def forward(self, x):
        return self.fc(x)
