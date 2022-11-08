import warnings

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver
from models.GRU import GRU
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.metric import MAPE
from utils.plot_results import plot_results
from utils.utils import adjust_learning_rate, EarlyStopping


class GRUSolver(BasicSolver):
    def __init__(self, args):
        super(GRUSolver, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = GRU(
            input_dim=args.enc_in,
            hidden_dim=args.d_model,
            output_dim=args.c_out * args.pred_len,
            num_layers=args.e_layers,
            dropout=args.dropout,
            use_batch_normalize=args.use_batch_normalize,
            use_layer_normalize=args.use_layer_normalize,
            seq_len=args.seq_len,
            use_BiLSTM=args.use_BiLSTM,
            last_num=args.last_num
        )

        if self.args.use_gpu and len(self.args.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.args.gpus)

        print("_build_model执行完毕，构建好GRU模型！")
        return model.to(self.device)

    def run_one_batch(self, batch_encoder_token):
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
