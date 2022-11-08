import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(NN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            # nn.Dropout(dropout),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        '''
        Input:
            (batch_size, seq_len, args.enc_in)
        Output:
            (batch_size, 1, args.c_out)
            dim=1代表pred_len;  LSTM只输出1个pred
        '''

        x = x.reshape(x.shape[0], -1)
        output = self.encoder(x)
        output = self.decoder(output)

        return output.unsqueeze(1)

class CEEMDANNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(CEEMDANNN, self).__init__()
        self.models = nn.ModuleDict({
            'LSTM_for_low_freq': NN(
                input_dim, hidden_dim, output_dim, dropout
            ),
            'LSTM_for_mid_freq': NN(
                input_dim, hidden_dim, output_dim, dropout
            ),
            'LSTM_for_high_freq': NN(
                input_dim, hidden_dim, output_dim, dropout
            ),
        })

    def forward(self, x):
        '''
            x: (batch_size, seq_len, feature_num * 3)
            outputs: (batch_size, seq_len, args.c_out * 4)
        '''
        feature_num = x.shape[-1] // 3
        outputs = []
        for i, (model_name, model) in enumerate(self.models.items()):
            output = self.models[model_name](x[:, :, feature_num * i:feature_num * (i + 1)])
            outputs.append(output)
        outputs = [outputs[0] + outputs[1] + outputs[2]] + outputs
        outputs = torch.cat(outputs, dim=-1)

        return outputs
