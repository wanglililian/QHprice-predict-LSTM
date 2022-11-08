import torch
import torch.nn as nn


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """

    def __init__(self, num_features, out_channels, kernel_size):
        super(CLSTM_cell, self).__init__()

        self.num_features = num_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # in this way the output has the same size
        self.padding = (kernel_size - 1) // 2

        self.conv1d = nn.Sequential(
            nn.Conv1d(num_features + out_channels, 4 * out_channels, kernel_size, stride=1, padding=self.padding),
            nn.GroupNorm(4 * out_channels // 32, 4 * out_channels)
        )

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        '''
        input: (batch_size, seq_len, feature_num, sub_seq_len)
        '''
        device = inputs.device
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        feature_num = inputs.size(2)
        sub_seq_len = inputs.size(3)
        out_channels = self.out_channels

        if hidden_state is None:
            hx = torch.zeros(batch_size, out_channels, sub_seq_len).to(device)
            cx = torch.zeros(batch_size, out_channels, sub_seq_len).to(device)
        else:
            hx, cx = hidden_state

        output_inner = []
        for index in range(seq_len):
            x = inputs[index, :, :, :]

            combined = torch.cat((x, hx), 1)
            gates = self.conv1d(combined)  # gates: S, num_features*4, H, W

            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(gates, out_channels, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy

        return torch.stack(output_inner), (hy, cy)


class Conv1dLSTM(nn.Module):
    def __init__(self, num_features, out_channels, kernel_size, hidden_dim, output_dim, sub_seq_len, dropout):
        super(Conv1dLSTM, self).__init__()

        self.encoder = torch.nn.ModuleList(
            [
                CLSTM_cell(num_features, out_channels, kernel_size),
                # CLSTM_cell(out_channels, out_channels // 2, kernel_size),
                # CLSTM_cell(out_channels // 2, out_channels // 4, kernel_size)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * sub_seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        '''
        Input:
            (seq_len, batch_size, feature_num, sub_seq_len)
        Output:
            (batch_size, 1, args.c_out)
            dim=1代表pred_len;  LSTM只输出1个pred
        '''

        for layer in self.encoder:
            ht, (hT, cT) = layer(x)
            x = self.dropout(ht)

        # hT: (batch_size, out_channels, seq_len
        output = self.decoder(hT)

        return output
