import warnings

import torch.nn as nn

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver
from models.Conv1dLSTM import Conv1dLSTM


class Conv1dLSTMSolver(BasicSolver):
    def __init__(self, args):
        super(Conv1dLSTMSolver, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = Conv1dLSTM(
            num_features=args.enc_in,
            out_channels=args.d_model,
            kernel_size=5,
            hidden_dim=args.d_model * 4,
            output_dim=args.c_out * args.pred_len,
            sub_seq_len=args.sub_seq_len,
            dropout=args.dropout
        )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def run_one_batch(self, batch_encoder_token, batch_decoder_token, batch_x_temporal, batch_y_temporal):
        '''
        Input:
            batch_encoder_token: (batch_size, seq_len, args.enc_in) -> (seq_len, batch_size, args.enc_in, sub_seq_len)
        Output:
            outputs:             (pred_len, batch_size, args.c_out)
        '''

        batch_size = batch_encoder_token.size(0)
        seq_len = batch_encoder_token.size(1)
        sub_seq_len = self.args.sub_seq_len
        feature_num = batch_encoder_token.size(2)

        x_input = batch_encoder_token.reshape(batch_size, seq_len // sub_seq_len, sub_seq_len, feature_num)
        x_input = x_input.permute(1, 0, 3, 2)
        outputs = self.model(
            x_input.to(self.device),
        )

        return outputs.view(outputs.shape[0], self.args.pred_len, self.args.c_out)
