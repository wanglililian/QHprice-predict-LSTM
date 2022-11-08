import warnings

import torch.nn as nn

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver
from models.EnsembleModel import EnsembleModel
import os
import torch


class EnsembleSolver(BasicSolver):
    def __init__(self, args):
        super(EnsembleSolver, self).__init__(args)

    def _build_model(self):
        args = self.args
        path1 = './checkpoints/LSTM/LSTM_id0711/01_Regression_class2_src=day_tgt=one_week_kalmanFalse_al0.005_sl200_ll1_pl1_dm2048_nh8_el2_dl1_df2048_atprob_eblearned_dr0.6_destest_ii0_t1626244933.4832845_LastNum1_ln1/checkpoint_all.pth'
        path2 = './checkpoints/LSTM/LSTM_id0711/01_Regression_class2_src=day_tgt=one_week_kalmanFalse_al0.005_sl200_ll1_pl1_dm2048_nh8_el2_dl1_df2048_atprob_eblearned_dr0.6_destest_ii0_t1626247106.1790872_LastNum5_ln1/checkpoint_all.pth'
        model = EnsembleModel(
            model_type_list=['LSTM_one_week_last1','LSTM_one_week_last5'],
            model_path_list=[path1,path2],
            seq_len_list=[200,200],
            weight_list=[0.1,0.9]
        ).cuda()

        if self.args.use_gpu and len(self.args.gpus) > 1:
            print('use multi gpus')
            model = nn.DataParallel(model, device_ids=self.args.gpus)
        print(self.args.gpus)
        print(self.device)
        return model.cuda()

    def run_one_batch(self, batch_encoder_token, batch_decoder_token, batch_x_temporal, batch_y_temporal):
        '''
        Input:
            batch_encoder_token: (batch_size, seq_len, args.enc_in)
        Output:
            outputs:             (batch_size, pred_len, args.c_out)
        '''


        outputs = self.model(
            batch_encoder_token.cuda(),
        )

        return outputs.view(outputs.shape[0], self.args.pred_len, self.args.c_out)
