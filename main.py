import argparse
import time

import torch

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solvers.LSTMSolver import LSTMSolver
from solvers.CEEMDANLSTMSolver import CEEMDANLSTMSolver
from solvers.CEEMDANNNSolver import CEEMDANNNSolver
from solvers.InformerSolver import InformerSolver
from solvers.NNSolver import NNSolver
from solvers.Conv1dLSTMSolver import Conv1dLSTMSolver
from solvers.EnsembleSolver import EnsembleSolver
from solvers.MultiLSTMSolver import MultiLSTMSolver
from solvers.MyLayerNormLSTMSolver import MyLayerNormLSTMSolver
from solvers.GRUSolver import GRUSolver

def get_args_parser():
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--model', type=str, default='NN',
                        help='model of experiment, options: [GRU, LSTM, NN, Conv1dLSTM, MultiLSTM]')

    parser.add_argument('--frequence', type=str, default='day', help='frequence [day]')
    parser.add_argument('--target_frequence', type=str, default='one_week', help='frequence [one_week,two_week, one_month]')  # 这里修改预测时长参数，包括one_week,two_week, one_month
    parser.add_argument('--normalize', default='', type=str, help='normalize', choices=['standard', 'minmax'])
    parser.add_argument('--kalman', default=False, action="store_true", help='kalman')
    parser.add_argument('--use_DWT', default=False, action="store_true", help='whether to use DWTT')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='final_lwg_2009-2022_Day_minmax.csv',
                        help='location of the data file')  # default='Latest_CL_Data/Day_Week_Month/WTI_1987-2021_Day.csv'
    parser.add_argument('--pre_data_path', type=str, default='./data/螺纹钢主力.xlsx',
                        help='location of the data file')
    parser.add_argument('--result_path', type=str, default='GRU_id0_Regression_class2_src=day_tgt=one_week_sl20_ii0_t1652615443.707003_LastNum1',
                        help='location of the data file')
    parser.add_argument('--task', type=str, default='Regression', help='task')
    parser.add_argument('--class_num', type=int, default=2, help='class num')

    parser.add_argument('--seq_len', type=int, default=20, help='input series length') # 这里修改输入序列的长度：one_week:20;one_month:60
    parser.add_argument('--sub_seq_len', type=int, default=20, help='input series sub length')
    parser.add_argument('--label_len', type=int, default=1, help='help series length')
    parser.add_argument('--pred_len', type=int, default=1, help='predict series length')
    parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')     # LSTM的输入特征数目
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')     # LSTM的输出长度
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=2048, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')  # only for informer
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    parser.add_argument('--dropout', type=float, default=0.6, help='dropout')
    parser.add_argument('--alpha', type=float, default=0.005, help='alpha')
    parser.add_argument('--last_num', type=int, default=1, help='last_num')
    parser.add_argument('--use_data_ratio', type=float, default=1, help=' use part of data for quick test')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='learned',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder',
                        default=False)
    parser.add_argument('--use_signal_loss', action='store_true', help='whether to use signal_loss in Regression task',
                        default=False)
    parser.add_argument('--signal_loss_weight', type=float, default=1.0, help='signal loss weight')
    parser.add_argument('--returns_scale_weight', type=float, default=1.0, help='returns_scale_weight')

    parser.add_argument('--use_batch_normalize', action='store_true', help='whether to use batch normalize',
                        default=False)
    parser.add_argument('--use_gradient_norm', action='store_true', help='whether to use gradient normalize',
                        default=False)
    parser.add_argument('--use_layer_normalize', action='store_true', help='whether to use layer normalize',
                        default=False)
    parser.add_argument('--use_mae_loss', action='store_true', help='whether to use mae_loss in Regression task',
                        default=True)
    parser.add_argument('--use_BiLSTM', action='store_true', help='whether to use BiLSTM',
                        default=False)
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='file list')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=150, help='train epochs')   # 这里修改训练的轮数
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data') # 这里修改batch size
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience') # 模型提前停止的patience
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')  # 这里修改学习率大小：0.0001/0.00001
    parser.add_argument('--weight_decay', type=float, default=0.1, help='optimizer weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer [SGD, Adam]')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='StepLR', help='adjust learning rate')
    parser.add_argument('--log_dir', type=str, default='tensorboard_logs', help='tensorboard log dirs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint dirs')

    parser.add_argument('--use_gpu', default=True, action="store_true", help='use gpu')  # 这里修改为使用cpu
    parser.add_argument('--use_multi_gpu', default=False, action="store_true", help='use gpu')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpus')
    parser.add_argument('--log_id', type=str, default='0', help='log_id')
    parser.add_argument('--test_only', action='store_true', help='whether to only test',
                        default=False)
    parser.add_argument('--use_three_freq', action='store_true', help='whether to use three frequence',
                        default=False)
    parser.add_argument('--use_multi_data', action='store_true', help='whether to use multi data',
                        default=False)
    return parser

def main():
    args= get_args_parser().parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    solvers_zoo = {
        'GRU':GRUSolver,
        'informer': InformerSolver,
        'LSTM': LSTMSolver,
        'NN': NNSolver,
        'Conv1dLSTM': Conv1dLSTMSolver,
        'Ensemble': EnsembleSolver,
        'CEEMDANLSTM': CEEMDANLSTMSolver,
        'CEEMDANNN': CEEMDANNNSolver,
        'MyLayerNormLSTM': MyLayerNormLSTMSolver,
        'MultiLSTM': MultiLSTMSolver
    }

    Solver = solvers_zoo[args.model]

    print('learning rate',args.learning_rate)

    for ii in range(args.itr):
        setting = '{}_id{}_{}_class{}_src={}_tgt={}_kalman{}_al{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_dr{}_des{}_ii{}_t{}'.format(
            args.model,
            args.log_id,
            args.task,
            args.class_num,
            args.frequence,
            args.target_frequence,
            str(args.kalman),
            args.alpha,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.attn,
            args.embed,
            args.dropout,
            args.des,
            ii,
            time.time())
        setting += ('_LastNum' + str(args.last_num))
        if args.use_layer_normalize:
            setting += '_ln1'
        if args.use_batch_normalize:
            setting += '_bn1'

        solver = Solver(args)

        if args.test_only:
            solver.test(setting)
        else:
            solver.train(setting)

        torch.cuda.empty_cache()

if __name__ == '__main__':
     main()