import torch.nn as nn

# 定义GRU模型
# 为了让模型更加规范化，本文将整个预测模型划分为Encoder-Decoder两部分
# Encoder就是Gru通过输入序列数据进行特征提取
# Decoder中定义了线性层实现结果的回归
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.3, use_batch_normalize=False, use_layer_normalize=False, \
                 seq_len=200, use_BiLSTM=False, last_num=1):
        super(GRU, self).__init__()

        # encoder定义，使用torch中预定义好的GRU模型
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=False,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=use_BiLSTM,
        )
        # 是否使用根据传入的参数判断是否使用batch正则化或者层正则化
        self.use_batch_normalize = use_batch_normalize
        self.use_layer_normalize = use_layer_normalize
        if self.use_batch_normalize:
            self.bn1 = nn.BatchNorm1d(input_dim)

        self.last_num = last_num

        input_dim_for_decoder = hidden_dim

        if use_layer_normalize:
            self.ln1 = nn.LayerNorm([seq_len, input_dim])
            self.ln2 = nn.LayerNorm([seq_len, input_dim_for_decoder])

        # decoder定义,定义了四个线性层Linear
        input_dim_for_decoder = input_dim_for_decoder * last_num
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim_for_decoder, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 16, output_dim)
        )

    def forward(self, x):
        '''
        Input:
            (batch_size, seq_len, args.enc_in)
        Output:
            (batch_size, 1, args.c_out)
            dim=1代表pred_len;  LSTM只输出1个pred
        '''
        # 根据参数对输入进行正则化
        if self.use_layer_normalize:
            x = self.ln1(x)

        if self.use_batch_normalize:
            x = x.permute(0, 2, 1)
            x = self.bn1(x)
            x = x.permute(0, 2, 1)

        # x维度的变化：(batch_size, seq_len, args.enc_in) -> (seq_len, batch_size, args.enc_in)
        x = x.permute(1, 0, 2)
        # 输入encoder
        ht, (hT, cT) = self.encoder(x)

        # 对encoder的输出的维度在变回来：(seq_len, batch_size, hidden_dim) - > (batch_size, seq_len, hidden_dim)
        ht = ht.permute(1, 0, 2)
        # if self.use_layer_normalize:
        #     ht = self.ln2(ht)

        batch_size = ht.shape[0]
        # 输入decoder获取输出
        output = self.decoder(ht[:, -self.last_num:, :].reshape(batch_size, -1))

        return output
