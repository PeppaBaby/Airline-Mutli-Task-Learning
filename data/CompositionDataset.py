import torch
from torch.utils.data import Dataset

from data.HistoryStatDataset import HistoryStatDataset
from data.TicketOrderDataset import TicketOrderDataset


##从当前订单中学习用于”票价等级预测“的类别标签
class CompositionDataset(Dataset):

    def __init__(self, stat_dir, ticket_dir):
        # 从stat_dir目录读取历史销售统计数据
        self.stat_set = HistoryStatDataset(stat_dir)
        # 从ticket_dir目录读取当前订单数据
        self.ticket_set = TicketOrderDataset(ticket_dir)

        # 更改TicketOrderDataset中的airlines，将它从字符串变为整数索引
        self.ticket_set.airlines = torch.LongTensor([self.stat_set.air2idx[air] for air in self.ticket_set.airlines])

    # override the method __getitem__ in the abstract class Dataset
    # 返回的批数据必须作为Model.forward函数的输入(hfs,ds,tfs)
    # hfs: [Batch,17,730], batch为批大小； 730为两年销售数据； 17为航班总共17个票价等级
    # ds: [Batch,] 当前批中每个订单的日期
    # Xs:[Batch,nfeatures] 当前批中每个订单的特征
    def __getitem__(self, index):
        # 从当前订单数据数据集中读取批
        airs, ds, Xs, Ys = self.ticket_set[index]
        # hfs: [Batch,17,730], batch为批大小； 730为两年销售数据； 17为航班总共17个票价等级
        # orign: 起飞机场
        # dest: 目的机场
        hfs, orign, dest = self.stat_set[airs]
        # 获取第d天各票价等级的销售数据，从hfs访问第ds列
        if hfs.ndim == 2:
            d_pred = hfs[:,ds]
        else:
            d_pred = torch.stack([m[:,d] for m,d in zip(hfs,ds)],axis = 0)

        return hfs, ds, Xs, Ys, d_pred, orign, dest

    # override the method __len__ in the abstract class Dataset
    def __len__(self):
        return len(self.ticket_set)




# 测试代码是否正常
if __name__ == '__main__':
    dataset = CompositionDataset(r'../airline/count',
                                 r'../airline/PEK_pro')
    print(len(dataset))
    print(dataset[:2])
