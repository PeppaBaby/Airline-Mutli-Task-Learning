import itertools
import numpy as np
import torch
from torch import nn


# 从历史销售数据中，利用1D卷积，提取时序特征
# 并将此时序特征作为当前订单的部分输入
class HistoryFeature(nn.Module):
    def __init__(self, input_shape, out_channels, ks, ds, output_size, dropout=0.1):
        """
        input_shape: [730*17], 前者为天数，后者为每天各票价等级的销售量
        out_channels: 1D 卷积核数
        ks: 卷积核大小, 或者多少天销售记录参与卷积, k= 3,5,7
        ds: 卷积核元素之间的间距，相隔多少天。比如1（连续天数）、7（相隔1周）、14（相隔两周）、30（相隔1月）
        output_size: 输出特征的温度（可以是票价等级数）
        """
        super(HistoryFeature, self).__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=input_shape[0],
                                                            out_channels=out_channels,
                                                            kernel_size=k, dilation=d),
                                                  nn.BatchNorm1d(num_features=out_channels),
                                                  nn.ReLU(),
                                                  nn.MaxPool1d(kernel_size=input_shape[0] - k + 1),
                                                  nn.Dropout(dropout)) for (k, d) in itertools.product(ks, ds)])

        # 经过上述卷积后，进行拼接，可以得到[Batch, out_channels, conv_size]
        conv_size = self._get_conv_size(input_shape)

        # 经过2个线性层变化
        self.fc = nn.Sequential(nn.Linear(out_channels * conv_size, 256),
                                nn.ReLU(),
                                nn.Linear(256, output_size),
                                nn.Dropout(dropout))

    # 得到卷积核的大小
    def _get_conv_size(self, shape):
        return np.sum([conv((torch.zeros(1, *shape))).size(-1) for conv in self.convs])

    def forward(self, x):
        """
        x: [Batch_size x  num_class x num_days]
        Batch_size: 一批中样本的个数
        num_class: 票价等级数，类似单词的embedding_size
        num_days: 每个批涉及多少天的销售记录，类似句子最大的长度
        feature_size: 卷积后的输出维度
        """
        out = torch.cat([conv(x) for conv in self.convs], dim=-1)

        return self.fc(out.view(x.size(0), -1).contiguous())
