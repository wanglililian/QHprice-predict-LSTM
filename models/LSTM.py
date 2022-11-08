import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.3, use_batch_normalize=False, use_layer_normalize=False, \
                 seq_len=200, use_BiLSTM=False, last_num=1):
        super(LSTM, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=False,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=use_BiLSTM,
        )

        self.use_batch_normalize = use_batch_normalize
        self.use_layer_normalize = use_layer_normalize

        if self.use_batch_normalize:
            self.bn1 = nn.BatchNorm1d(input_dim)

        self.last_num = last_num
        if use_BiLSTM:
            input_dim_for_decoder = hidden_dim * 2
        else:
            input_dim_for_decoder = hidden_dim

        if use_layer_normalize:
            self.ln1 = nn.LayerNorm([seq_len, input_dim])
            self.ln2 = nn.LayerNorm([seq_len, input_dim_for_decoder])

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

        if self.use_layer_normalize:
            x = self.ln1(x)

        if self.use_batch_normalize:
            x = x.permute(0, 2, 1)
            x = self.bn1(x)
            x = x.permute(0, 2, 1)

        # (batch_size, seq_len, args.enc_in) -> (seq_len, batch_size, args.enc_in)
        x = x.permute(1, 0, 2)
        ht, (hT, cT) = self.encoder(x)

        # (seq_len, batch_size, hidden_dim) - > (batch_size, seq_len, hidden_dim)
        ht = ht.permute(1, 0, 2)
        # if self.use_layer_normalize:
        #     ht = self.ln2(ht)

        batch_size = ht.shape[0]
        output = self.decoder(ht[:, -self.last_num:, :].reshape(batch_size, -1))

        return output
