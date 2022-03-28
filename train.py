import logging
import os
import time
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from data.CompositionDataset import CompositionDataset
from embedding.HistoryEmbedding import HistoryEmbedding
from embedding.TicketEmbedding import TicketEmbedding
from modules.DemandPredictionModel import DemandPredictionModel
from modules.HistoryFeature import HistoryFeature
from modules.MultiTaskModel import MultiTaskModel
from modules.TicketFeature import TicketFeature
import numpy as np
from sklearn import metrics

def make_model(hist_matrix_shape, out_channels, kernel_size, dilation_size,
               nfeatures, embeded_size, orign_vocab, dest_vocab,
               d_model, n_classes,n_layers=2, dropout=0.1):
    """
    hist_matrix_shape: (17,730)； 730为两年销售数据； 17为航班总共17个票价等级
    out_channels: 历史销售矩阵的输出通道，设置1，可以调节
    kernel_size: 卷积核的大小，多少个元素进行卷积，分别设置为[3,5,7]
    dilation_size：相隔多少个元素，[1,7,14,30],分别对应着天、周、半月、1个月
    nfeatures: 原始订单中，每维特征有多少个离散值，它是由数据集确定
    embeded_size: 每个特征映射的大小，设置为32，也可以调节
    orign_vocab: 起飞机场数
    dest_vocab： 目的机场数
    d_model: 隐藏层大小，可以理解为原始特征经过变化后的大小，也可以调节
    n_classes: 总共有多少类，数据集17类
    n_layers: 将卷积网络+原始订单网络的结果，再经过2层（layer norm + Residual connection 网络）
    dropout： dropout 概率
    """

    # 卷积网络：学习历史定价数据的时序特征
    hist_model = nn.Sequential(HistoryEmbedding(hist_matrix_shape),
                               HistoryFeature(hist_matrix_shape, out_channels, kernel_size, dilation_size, d_model,
                                              dropout))

    # NN网络：学习当前订单的特征
    ticket_model = nn.Sequential(TicketEmbedding(nfeatures, embeded_size),
                                 TicketFeature(len(nfeatures) * embeded_size, d_model))
    # NN网络：利用第1,2.., d-1天的时序特征，预测第d天
    demand_predictor = DemandPredictionModel(nn.Embedding(orign_vocab, embeded_size),
                                             nn.Embedding(dest_vocab, embeded_size),
                                             n_layers, d_model + 2 * embeded_size, dropout, n_classes)

    # 联合机票等级预测模型、需求量预测模型
    model = MultiTaskModel(hist_model, ticket_model, demand_predictor, n_layers, d_model, dropout, n_classes).cuda(
        device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def load_logger(output_dir, coeffient, index):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}:coeffient_{}_repeats_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'),coeffient, index)
    final_log_file = os.path.join(output_dir, log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


# 测试代码是否正常
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 此路径为本地的绝对路径
    dataset = CompositionDataset(r'/media/test/C/code_ed/comp1/exe_data/count', #航线需求数据集
                                 r'/media/test/C/code_ed/comp1/exe_data/one_year') #订单数据集

    # 设置不同的多目标损失的组合系数，以同时优化需求预测损失、多票价等级预测损失
    # 设置不同的多任务权重参数：0.001, 0.002, 0.005, 0.01, 0.02, 0.05
    for coeffient in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
        # 相同系数运行5次
        for iter in range(1):
            # 80%数据用于训练，20%数据用于测试
            train_size = int(len(dataset) * 0.8) #测试集为80%
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

            #构建train_loader，批次1024，shuffle，设置8个线程
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True,drop_last=True,pin_memory=True,num_workers=8)
            #构建 test_loader
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True,drop_last=True,pin_memory=True,num_workers=8)
            #生成模型
            # def make_model(hist_matrix_shape, out_channels, kernel_size, dilation_size,
            #                nfeatures, embeded_size, orign_vocab, dest_vocab,
            #                d_model, n_classes,n_layers=2, dropout=0.1)
            # [1, 3, 5, 7]卷积核； [1 ,7, 28]卷积步长
            model = make_model((17, 730), 1, [1, 3, 5, 7], [1 ,7, 28],
                               dataset.ticket_set.nfeatures, 32, dataset.stat_set.origin_vocab,dataset.stat_set.dest_vocab,
                               64, 17, n_layers=2, dropout=0.1).to(device)

            logger = load_logger('./log', coeffient, iter)
            logger.info('start training!')
            time_start = time.time()
            # 2个损失函数：票价预测损失和需求预测损失
            ticket_pred_loss_func = nn.NLLLoss()
            demand_pred_loss_func = nn.MSELoss()
            #构建adam优化器
            optimizer = optim.Adam(model.parameters(),lr=0.01)

            # 每个模型训练训练1000次
            for epoch in range(1000):
                # 设置模型为训练模式
                time_start = time.time()
                model.train()
                # 每一次损失为train_loss
                train_loss,train_acc = 0,0.0
                # 每次读取一个batch数据，其中
                # hfs: [Batch,17,730], batch为批大小； 730为两年销售数据； 17为航班总共17个票价等级
                # ds: [Batch],每个订单涉及的日期
                # Xs: [Batch,nfeatures] 当前批中每个订单的特征
                # Ys: [Batch], 每个订单的票价等级
                # d_pred: 第d天的需求
                # orign: 起飞机场
                # dest: 目的机场
                for step, data in enumerate(train_loader):
                    # 把批数据从CPU迁移到GPU
                    hfs, ds, Xs, Ys, d_pred, origns, dests = [d.to(device) for d in data]
                    # 调用多任务模型，两个输出
                    ticket_predicted, demand_predicted = model(hfs, ds, Xs, origns, dests)
                    # 得到多任务合并损失,由于座位数都是100以上计量
                    loss = ticket_pred_loss_func(ticket_predicted, Ys) + \
                           coeffient / 10 * demand_pred_loss_func(demand_predicted, d_pred)
                    # 多任务模型的梯度优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 总训练损失
                    train_loss += loss.item()
                    # train_acc += torch.sum(torch.argmax(ticket_predicted, 1) == Ys).item()
                # 每训练10次，就预测1次
                if epoch % 10 == 0:

                    test_loss,test_acc = 0,0.0
                    # 设置为测试模式
                    model.eval()
                    predict_all = np.array([], dtype=float)
                    labels_all = np.array([], dtype=float)
                    # 分批读取测试数据
                    for data in test_loader:
                        # 把批数据从CPU迁移到GPU
                        hfs, ds, Xs, Ys, d_pred, origns, dests = [d.to(device) for d in data]
                        # 调用多任务模型，两个输出
                        ticket_predicted, _ = model(hfs, ds, Xs, origns, dests)
                        # 关注多分类损失
                        test_loss += loss.item()
                        #准确率
                        test_acc += torch.sum(torch.argmax(ticket_predicted, 1) == Ys).item()
                        ys = Ys.data.cpu().numpy()
                        pre_y = torch.argmax(ticket_predicted.data, 1).cpu().numpy()
                        labels_all = np.append(labels_all, ys)
                        predict_all = np.append(predict_all, pre_y)
                        target_name = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                    #采用宏观平均（精确率、召回率、F1）和微观平均（精确率、召回率、F1）两种评价指标
                    report = metrics.classification_report(labels_all, predict_all, target_name, digits=4)
                    time_end = time.time()
                    time_c = time_end - time_start  # 运行所花时间
                    logger.info('Epoch:[{}]\t train_loss={:.5f}\t test_loss={:.5f}\t time_cost={:.5f}s\t'.format((epoch/10+1),
                                train_loss/len(train_dataset), test_loss/len(test_dataset), time_c))
                    logger.info('Epoch:[{}] \n {}\t'.format((epoch/10+1),report))
            time_end = time.time()
            time_c = time_end - time_start  # 运行所花时间
            print('time cost', time_c, 's')
            logger.info(model)
            logger.info('finish training!')
