import warnings

import torch.nn as nn

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver
from models.NN import NN


class NNSolver(BasicSolver):
    def __init__(self, args):
        super(NNSolver, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = NN(
            input_dim=args.enc_in * args.seq_len,
            hidden_dim=args.d_model,
            output_dim=args.c_out * args.pred_len,
            dropout=args.dropout
        )

        if self.args.use_multi_gpu and self.args.use_gpu:
            print('use multi gpu')
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def run_one_batch(self, batch_encoder_token, batch_decoder_token, batch_x_temporal, batch_y_temporal):
        '''
        Input:
            batch_encoder_token: (batch_size, seq_len, args.enc_in)
        Output:
            outputs:             (batch_size, pred_len, args.c_out)
        '''

        outputs = self.model(
            batch_encoder_token.to(self.device),
        )

        return outputs.view(outputs.shape[0], self.args.pred_len, self.args.c_out)
