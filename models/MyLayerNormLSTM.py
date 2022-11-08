import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyLayerNormLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.3,
                 use_batch_normalize=False, use_layer_normalize=False,
                 seq_len=200, use_BiLSTM=False, last_num=1):
        super(MyLayerNormLSTM, self).__init__()

        self.use_batch_normalize = use_batch_normalize
        self.use_layer_normalize = use_layer_normalize

        if self.use_batch_normalize:
            self.bn1 = nn.BatchNorm1d(input_dim)

        self.last_num = last_num
        self.encoder = LayerNormLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            bidirectional=use_BiLSTM,
            use_layer_norm=use_layer_normalize
        )
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


class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True, use_layer_norm=True):
        super().__init__(input_size, hidden_size, bias)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
            self.ln_ho = nn.LayerNorm(hidden_size)
        # DropConnect on the recurrent hidden to hidden weight
        self.dropout = dropout

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        weight_hh = nn.functional.dropout(self.weight_hh, p=self.dropout, training=self.training)
        if self.use_layer_norm:
            gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                    + self.ln_hh(F.linear(hx, weight_hh, self.bias_hh))
        else:
            gates = F.linear(input, self.weight_ih, self.bias_ih) \
                    + F.linear(hx, weight_hh, self.bias_hh)

        i, f, c, o = gates.chunk(4, 1)
        i_ = torch.sigmoid(i)
        f_ = torch.sigmoid(f)
        c_ = torch.tanh(c)
        o_ = torch.sigmoid(o)
        cy = (f_ * cx) + (i_ * c_)
        if self.use_layer_norm:
            hy = o_ * self.ln_ho(torch.tanh(cy))
        else:
            hy = o_ * torch.tanh(cy)
        return hy, cy


class LayerNormLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 weight_dropout=0.0,
                 bias=True,
                 bidirectional=False,
                 use_layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # using variational dropout
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
                for layer in range(num_layers)
            ])

    def forward(self, input, hidden=None, seq_lens=None):
        device = input.device
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = []
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_size, requires_grad=False)
        if seq_lens != None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        else:
            seq_lens = [seq_len for i in range(batch_size)]
        seq_len_mask = seq_len_mask.transpose(0, 1)

        if self.bidirectional:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_ = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size]).to(device)
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_reverse = torch.LongTensor([0] * batch_size).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size]).to(device)
            indices = torch.cat((indices_, indices_reverse), dim=1)
            hy = []
            cy = []
            xs = input
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, 2, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, 2, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht_, ct_ = layer0(x0, (h0, c0))
                    ht[t][l0] = ht_ * seq_len_mask[t]
                    ct[t][l0] = ct_ * seq_len_mask[t]
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht_, ct_ = layer1(x1, (h1, c1))
                    ht[t][l1] = ht_ * seq_len_mask[t]
                    ct[t][l1] = ct_ * seq_len_mask[t]
                    h1, c1 = ht[t][l1], ct[t][l1]

                xs = [torch.cat((h[l0] * dropout_mask[l][0], h[l1] * dropout_mask[l][1]), dim=1) for h in ht]
                ht_temp = torch.stack([torch.stack([h[l0], h[l1]]) for h in ht])
                ct_temp = torch.stack([torch.stack([c[l0], c[l1]]) for c in ct])
                if len(hy) == 0:
                    hy = torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    hy = torch.cat((hy, torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
                if len(cy) == 0:
                    cy = torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    cy = torch.cat((cy, torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
            y = torch.stack(xs)
        else:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, self.num_layers, 1, self.hidden_size]).to(device)
            h, c = hx, cx
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht_, ct_ = layer(x, (h[l], c[l]))
                    ht[t][l] = ht_ * seq_len_mask[t]
                    ct[t][l] = ct_ * seq_len_mask[t]
                    x = ht[t][l] * dropout_mask[l]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] * dropout_mask[-1] for h in ht]).to(device)
            hy = torch.stack(list(torch.stack(ht).gather(dim=0, index=indices).squeeze(0))).to(device)
            cy = torch.stack(list(torch.stack(ct).gather(dim=0, index=indices).squeeze(0))).to(device)

        return y, (hy, cy)
