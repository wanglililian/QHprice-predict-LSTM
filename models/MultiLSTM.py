import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, use_layer_normalize, seq_len,
                 use_BiLSTM):
        super(MyLSTM, self).__init__()
        self.use_layer_normalize = use_layer_normalize
        if use_layer_normalize:
            self.ln1 = nn.LayerNorm([seq_len, input_dim])

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=False,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=use_BiLSTM,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        # self.decoder = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, hidden_dim // 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 4, hidden_dim // 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 4, output_dim)
        # )

    def forward(self, input):
        '''
        Input:
            (batch_size, seq_len, args.enc_in)
        Output:
            (batch_size, 1, args.c_out)
            dim=1代表pred_len;  LSTM只输出1个pred
        '''
        if self.use_layer_normalize:
            input = self.ln1(input)

        input = input.permute(1, 0, 2)  # (batch_size, seq_len, args.enc_in) -> (seq_len, batch_size, args.enc_in)
        ht, (hT, cT) = self.encoder(input)  # hT: (num_layers * num_directions, batch_size, hidden_dim)

        hT = hT[-1]  # hT: (batch_size, hidden_dim)

        output = self.decoder(hT).unsqueeze(1)  # output: (batch_size, 1, output_dim)

        return output


class MultiLSTM(nn.Module):
    def __init__(self, cnt, input_dim, hidden_dim, output_dim, num_layers, dropout, use_layer_normalize, seq_len,
                 use_BiLSTM):
        super(MultiLSTM, self).__init__()

        self.cnt = cnt
        self.multi_lstms = nn.ModuleList([
            MyLSTM(
                input_dim, hidden_dim, output_dim, num_layers, dropout, use_layer_normalize, seq_len, use_BiLSTM
            ) for i in range(cnt)
        ])


    def forward(self, x, stage):
        '''
            Input:
                (batch_size, seq_len, args.enc_in)
            Output:
                (batch_size, 1, args.c_out)
                dim=1代表pred_len;  LSTM只输出1个pred
        '''

        output = None
        for i in range(stage):
            if output is None:
                output = self.multi_lstms[i](x)
            else:
                output += self.multi_lstms[i](x)

        return output
