import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Dataset.CLDataset import CLDataset
from utils.automatic_weighted_loss import AutomaticWeightedLoss
from utils.metric import MAPE, MEDIAN_MAPE
from utils.plot_results import plot_results
from utils.utils import adjust_learning_rate, EarlyStopping


# 将DataLoader的创建、模型的训练以及测试封装
class BasicSolver(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.history_grad_norm = []
        self.model_name = args.model
        self.model = self._build_model()  # 此处创建model，其子类会改写该函数，用于创建不同的模型
        self.log_dir = os.path.join(args.log_dir, self.model_name)
        self.root_path = args.root_path
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_name)
        self.writer = None

        self.dataset = {}
        self.scaled_dataset = {}
        self.data_loader = {}
        self.prepare_dataset()  # 调用准备数据集的函数，加载处理好的csv文件，构造dataset，并且封装成dataloader
        print("数据加载完毕！")

        self.awl = AutomaticWeightedLoss(num=2)
        self.criterion = self._select_criterion()
        self.model_optimizer = self._select_optimizer()

        print(self.model)

    def prepare_dataset(self):  # 该函数用于构建训练集、验证集、测试集
        args = self.args

        # 1、首先加载处理好的csv数据
        main_df = pd.read_csv(
            os.path.join(args.root_path, args.data_path),
            header=0,
            index_col='Date'
        )

        # 2、然后划分数据集
        self.dataset['shuffled_train'] = self._get_dataset(main_df['2009':'2018'], natural_week=False)
        self.dataset['test'] = self._get_dataset(main_df['2018':'2020'], natural_week=False)
        self.dataset['val'] = self._get_dataset(main_df['2020':], natural_week=False)

        self.data_loader['shuffled_train'] = self._get_dataloader(self.dataset['shuffled_train'], shuffle=True)
        self.data_loader['val'] = self._get_dataloader(self.dataset['val'], shuffle=False)
        self.data_loader['test'] = self._get_dataloader(self.dataset['test'], shuffle=False)

    def _get_dataset(self, df, natural_week=False):
        args = self.args

        dataset = CLDataset(
            df=df,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            use_data_ratio=args.use_data_ratio,
            frequence=args.frequence,
            target_frequence=args.target_frequence,
            natural_week=natural_week,
        )

        return dataset

    def _get_dataloader(self, dataset, shuffle, drop_last=False):
        args = self.args
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

    # 构建模型实例，该函数在子类中被覆盖
    def _build_model(self):
        raise NotImplementedError

    # 训练一个batch的函数，该函数在子类中已经被覆盖
    def run_one_batch(self, batch_encoder_token):
        raise NotImplementedError

    # 选择用于训练的设备是cpu还是gpu
    def _acquire_device(self):
        if self.args.use_gpu:
            devices = 'cuda:' + str(self.args.gpus[0])
            print('device is {}'.format(devices))
            device = torch.device(devices)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    # 该模型用于选择优化器
    # 根据参数args.optimizer控制，选择'Adam'和'SGD'
    def _select_optimizer(self):
        if self.args.optimizer == 'Adam':
            optimizer = optim.AdamW(list(self.model.parameters()) + list(self.awl.parameters()),
                                    weight_decay=self.args.weight_decay,
                                    lr=self.args.learning_rate)
        elif self.args.optimizer == 'SGD':
            optimizer = optim.SGD(list(self.model.parameters()) + list(self.awl.parameters()),
                                  lr=self.args.learning_rate, momentum=0.9,
                                  weight_decay=self.args.weight_decay)

        return optimizer

    # 该函数返回模型所使用的loss函数
    # 通过args.use_mae_loss参数控制选择，nn.L1Loss()与 nn.MSELoss()，选择其中一个
    def _select_criterion(self):
        criterion = None
        if self.args.task == 'Regression':
            if self.args.use_mae_loss:
                criterion = nn.L1Loss()
            else:
                criterion = nn.MSELoss()

        return criterion

    # 该函数用于计算模型的相关评估指标，参数type表示在什么数据上计算，包括“val”，“test”，“shuffled_train”
    def validate(self, type='val', epoch=0):

        args = self.args
        self.model.eval()  # 模型设置为评估模式
        total_loss = []

        cur_prices, predictions, trues, extended_tokens = [], [], [], []
        for batch_id, (batch_encoder_token, pred_token, extended_token) in enumerate(self.data_loader[type]):
            outputs = self.run_one_batch(batch_encoder_token)  # 计算模型预测值

            batch_label = pred_token.to(self.device)  # 标签

            mse_loss = self.criterion(  # 计算loss，此处采用的是L1loss
                outputs.reshape(-1),
                batch_label.reshape(-1)
            )
            loss = mse_loss

            total_loss.append(loss.item())
            extended_tokens.append(extended_token[:, -1, :].unsqueeze(1).detach())
            predictions.append(outputs.detach().cpu())
            trues.append(batch_label.detach().cpu())

        predictions = np.array(torch.cat(predictions, dim=0))
        trues = np.array(torch.cat(trues, dim=0))
        extended_tokens = np.array(torch.cat(extended_tokens, dim=0))

        mean_of_mape, median_of_mape = self.add_metrics(extended_tokens, predictions, trues, epoch, type)

        total_loss = np.average(total_loss)
        self.writer.add_scalar('Loss/{}'.format(type), total_loss, epoch)

        return total_loss, mean_of_mape, median_of_mape

    def train(self, log_name):
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(log_name))

        # 模型存储路径定义
        save_path = os.path.join(self.checkpoint_dir, log_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存系统当前时间，用于计算训练时间的参数
        time_now = time.time()

        # 定义一些训练需要的参数
        train_steps = len(self.data_loader['shuffled_train'])  # 训练一个epoch的step的数目
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 定义提前停止训练实例
        max_val_acc = -999999
        max_val_f1_use_val_f1 = -999999
        min_val_loss = 9999999

        # 开始训练
        for epoch in range(self.args.train_epochs):
            print('\n', '*' * 88, '\n')

            self.model.train()
            iter_count = 0
            train_loss = []  # 存储训练损失，后续可以用来作图，保存数据等

            for i, (batch_encoder_token, pred_token, extended_token) in enumerate(self.data_loader[
                                                                                      'shuffled_train']):  # 此处参数这么长不用管，我们的模型只用到了batch_encoder_token作为gru的输入；pred_token作为标签
                cur_batch_size = batch_encoder_token.shape[0]
                iter_count += 1

                '''
                    batch_encoder_token: (batch_size, seq_len, feature_dim = args.enc_in)
                    pred_token: (batch_size, pred_len, feature_dim)
                '''
                # 1、模型进行预测，预测结果为outputs
                outputs = self.run_one_batch(batch_encoder_token)  # 此处执行对应模型（lstm，gru）的run_one_batch函数

                '''
                    outputs: (batch_size, args.pred_len, args.c_out) 模型输出
                    batch_label: (batch_size, args.pred_len, 1)       对应输入数据的标签
                '''
                # 获取数据的真实标签batch_label
                batch_label = pred_token.to(self.device)

                # 2、损失函数计算loss
                mse_loss = self.criterion(
                    outputs.reshape(-1),
                    batch_label.reshape(-1)
                )
                loss = mse_loss

                # 保存loss
                train_loss.append(loss.item())

                # 3、反向传播，优化
                self.model_optimizer.zero_grad()
                loss.backward()  # loss反向传播
                self.model_optimizer.step()  # 优化器进行优化

                # 输出 训练的轮数、loss值、训练速度等，默认20steps输出一次
                if (i + 1) % 20 == 0:
                    print("\t iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, np.mean(train_loss)))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (train_steps - i)
                    print('\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            ##############   每个epoch训练结束后，计算当前的测试集、验证集上的一些结果    ####################################

            # 模型训练中间结果存储路径的定义
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, log_name))
            self.result_path = os.path.join(self.log_dir, log_name)

            # 每个epoch结束后都调用validate计算一下训练集、测试集、验证集上的相关参数
            train_loss, mean_of_mape, median_of_mape = self.validate(type='shuffled_train', epoch=epoch)  # 2004-2014
            val_loss, mean_of_mape, median_of_mape = self.validate(type='val', epoch=epoch)  # 2014-2017
            test_loss, mean_of_mape, median_of_mape = self.validate(type='test', epoch=epoch)  # 2017-2021

            print("\n Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss, test_loss))

            adjust_learning_rate(self.model_optimizer, epoch + 1, self.args)  # 调整学习率

            self.writer.add_scalar('learning rate', self.model_optimizer.param_groups[0]['lr'], epoch)

        best_model_path = save_path + '/' + 'checkpoint.pth'  # 保存最好的模型
        torch.save(self.model.state_dict(), best_model_path)  # 加载最好的模型
        self.writer.close()  # 关闭输出流
        return self.model

    def test(self, log_name):
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'EnsembleTest'))

        epoch = self.args.train_epochs - 1

        self.model.eval()
        # 加载训练好的模型进行test
        save_path = os.path.join(self.checkpoint_dir, self.args.result_path)
        best_model_path = save_path + '/' + 'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path,map_location='cpu'))

        self.result_path = os.path.join(self.args.root_path, log_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        test_loss, test_acc, test_f1 = self.validate(type='test', epoch=epoch)

        print('test acc = {}, test f1 = {}'.format(test_acc, test_f1))
        print("测试的生成结果保存在了:",self.result_path)

    def add_metrics(self, extended_info, preds, trues, epoch, type):
        '''
        input:
            extended_info: (batch_num, 1, extended_feature_num )    # feature_num = args.enc_in
            preds:      (batch_num, pred_len, feature_num = args.c_out)
            trues:      (batch_num, pred_len, feature_num = args.c_out)
        '''

        args = self.args
        print('\n', type)

        print('preds.shape = {}, trues.shape = {}'.format(preds.shape, trues.shape))
        # use only the first prediction
        preds = preds[:, 0, 0].reshape(-1)
        trues = trues[:, 0, 0].reshape(-1)
        Year = extended_info[:, 0, 1].reshape(-1)
        Month = extended_info[:, 0, 2].reshape(-1)
        Day = extended_info[:, 0, 3].reshape(-1)
        Weekday = extended_info[:, 0, 4].reshape(-1)

        pred_prices = preds
        true_prices = trues

        # 讲数据反向归一化回去
        pre_min, pre_max, next_min, next_max = inv_close_mean(args)
        # print(pre_min, pre_max, next_min, next_max)

        pred_prices = pred_prices * (next_max - next_min) + next_min
        true_prices = true_prices * (next_max - next_min) + next_min

        # 计算预测价格的损失函数包括，MAPE，median_MAPE
        mean_of_mape = MAPE(pred_prices, true_prices)
        median_of_mape = MEDIAN_MAPE(pred_prices, true_prices)
        self.writer.add_scalar('mean of mape/{}'.format(type), mean_of_mape, epoch)
        self.writer.add_scalar('median of mape/{}'.format(type), median_of_mape, epoch)

        # 最终预测结果数据保存在final_result_df中，有如下数据
        final_result_df = pd.DataFrame(
            data={
                'pred_next_prices': pred_prices,
                'true_next_prices': true_prices,
                'Year': Year,
                'Month': Month,
                'Day': Day,
                'WeekDay': Weekday
            }
        )
        final_result_df['mean_of_mape'] = mean_of_mape
        final_result_df['median_of_mape'] = median_of_mape

        # 绘制pred_prices, true_prices的图像
        figs = plot_results(pred_prices, true_prices)

        for i, fig in enumerate(figs):
            self.writer.add_figure(tag='{}/fig{}'.format(type, i), figure=fig, global_step=epoch)

        if type == 'test':
            final_result_df.to_csv(os.path.join(self.result_path, 'preds_test_epoch{}.csv'.format(epoch)))

        print("mean_of_mape of pred_price:", mean_of_mape)
        print("median_of_mape of pred_price:", median_of_mape)
        return mean_of_mape, median_of_mape


def inv_close_mean(args):
    df = pd.read_excel(
        args.pre_data_path,
        usecols=['日期', '收盘', '开盘', '最高', '最低']
    )

    df.columns = ['Date', 'Close', 'Open', 'High', 'Low']
    df.Date = pd.to_datetime(df.Date)
    df = df.sort_values(by=['Date'])
    df = df.set_index('Date', inplace=False).dropna()

    df['Returns'] = df['Close'].diff() / (df['Close'].shift(1))

    df = df.dropna()

    # 滚动
    for week_num in [1, 2, 4]:
        df['Pre_{}_Week_Mean'.format(week_num)] = df['Close'].rolling(window=week_num * 5).mean().shift(periods=1)
        df['Next_{}_Week_Mean'.format(week_num)] = df['Close'].rolling(window=week_num * 5).mean().shift(
            periods=-week_num * 5 + 1)
        df['Next_{}_Week_Mean_Diff'.format(week_num)] = df['Next_{}_Week_Mean'.format(week_num)] - df[
            'Pre_{}_Week_Mean'.format(week_num)]
        df['Next_{}_Week_Mean_Returns'.format(week_num)] = df['Next_{}_Week_Mean_Diff'.format(week_num)] / df[
            'Pre_{}_Week_Mean'.format(week_num)]

    if args.target_frequence == 'one_week':
        week_num = 1
    elif args.target_frequence == 'two_week':
        week_num = 2
    elif args.target_frequence == 'one_month':
        week_num = 4

    return df['Pre_{}_Week_Mean'.format(week_num)].min(), df['Pre_{}_Week_Mean'.format(week_num)].max(), df[
        'Next_{}_Week_Mean'.format(week_num)].min(), df['Next_{}_Week_Mean'.format(week_num)].max()
