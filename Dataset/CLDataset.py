import warnings

import numpy as np
import pandas as pd
import torch
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


class CLDataset(Dataset):
    def __init__(self, frequence='day', target_frequence='one_week',
                 seq_len=100, label_len=10, pred_len=10,
                 df=None, use_data_ratio=1.0, natural_week=False):

        self.seq_len = seq_len  # 序列的长度
        self.label_len = label_len  # 标签的长度
        self.pred_len = pred_len  # 预测的长度
        self.use_data_ratio = use_data_ratio

        self.frequence = frequence  # 使用的数据的频率，有天级的、分钟级的、等等，本模型就默认是“day“，用天的数据做预测
        self.target_frequence = target_frequence  # 要预测的是什么数据，本代码可以预测'one_week'、'two_week'、'one_month'的结果
        self.sample_indexes = []  # 用于存储可以使用的数据的下标

        self.df = df  # 就是读取的数据
        self.__read_data__()

    def __read_data__(self):
        # TODO 1: 读入
        df = self.df.dropna()  # 删除数据中的NAN
        # print(df.head())

        # TODO 2: 选择要使用的特征
        x_used_columns = ['Close', 'Open', 'Low', 'High']
        # x_used_columns = ['Close', 'Open', 'Low', 'High', 'Returns', 'Amplitude']  # x表示预测使用的特征对应的列
        z_used_columns = []  # z表示
        y_used_columns = []  # y表示标签使用的数据列

        if self.target_frequence == 'one_week':
            y_used_columns = ['Next_1_Week_Mean']
            z_used_columns = ['Pre_1_Week_Mean', 'Year', 'Month', 'Day', 'Weekday']
        elif self.target_frequence == 'two_week':
            y_used_columns = ['Next_2_Week_Mean_Returns']
            z_used_columns = ['Pre_2_Week_Mean', 'Year', 'Month', 'Day', 'Weekday']
        elif self.target_frequence == 'one_month':
            y_used_columns = ['Next_4_Week_Mean_Returns']
            z_used_columns = ['Pre_4_Week_Mean', 'Year', 'Month', 'Day', 'Weekday']
        elif self.target_frequence == 'one_day':
            y_used_columns = ['Returns']
            z_used_columns = ['Close', 'Year', 'Month', 'Day', 'Weekday']

        for i in range(0, len(df) - self.seq_len - self.pred_len - 1):
            self.sample_indexes.append(i)


        self.token_data_x = df[x_used_columns].values
        self.token_data_y = df[y_used_columns].values
        self.token_data_z = df[z_used_columns].values


    def __getitem__(self, index):
        '''
        output:
            encoder_token:    (seq_len, feature_dim = args.enc_in)                  # encoder的非时间类的输入数据
            pred_token:       (pred_len, 1)                                         # pred_len长度的标签
        '''
        index = self.sample_indexes[index]

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        encoder_token = self.token_data_x[s_begin:s_end]
        pred_token = self.token_data_y[r_end - self.pred_len:r_end]  # 标签
        extended_token = self.token_data_z[s_begin:s_end]

        return torch.FloatTensor(encoder_token),  torch.FloatTensor(pred_token), torch.FloatTensor(extended_token)

    def __len__(self):
        return len(self.sample_indexes)


# 设置pandas输出数据的格式，该函数可以忽略不看
def setup_pandas():
    # 显示所有列
    pd.set_option('display.max_columns', 200)

    # 显示所有行
    pd.set_option('display.max_rows', None)

    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 1000)
    pd.set_option('expand_frame_repr', False)


