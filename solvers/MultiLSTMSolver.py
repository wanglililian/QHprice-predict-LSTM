import warnings

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.utils import adjust_learning_rate, EarlyStopping
from models.MultiLSTM import MultiLSTM


class MultiLSTMSolver(BasicSolver):
    def __init__(self, args):
        super(MultiLSTMSolver, self).__init__(args)
        self.stage = 1

    def _build_model(self):
        args = self.args
        model = MultiLSTM(
            cnt=2,
            input_dim=args.enc_in,
            hidden_dim=args.d_model,
            output_dim=args.c_out * args.pred_len,
            num_layers=args.e_layers,
            dropout=args.dropout,
            use_layer_normalize = args.use_layer_normalize,
            seq_len = args.seq_len,
            use_BiLSTM=args.use_BiLSTM
        )

        if self.args.use_gpu and len(self.args.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.args.gpus)

        return model.to(self.device)

    def run_one_batch(self, batch_encoder_token, batch_decoder_token=None, batch_encoder_temporal=None, batch_decoder_temporal=None):
        '''
        Input:
            batch_encoder_token: (batch_size, seq_len, args.enc_in)
        Output:
            outputs:             (batch_size, pred_len, args.c_out)
        '''

        outputs = self.model(
            x=batch_encoder_token.to(self.device),
            stage=self.stage
        )

        return outputs.view(outputs.shape[0], self.args.pred_len, self.args.c_out)

    def train(self, log_name):
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(log_name))

        save_path = os.path.join(self.checkpoint_dir, log_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        time_now = time.time()

        train_steps = len(self.data_loader['shuffled_train'])
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        max_val_acc = -999999
        max_val_f1_use_val_f1 = -999999
        min_val_loss = 9999999
        for epoch in range(self.args.train_epochs):
            print('\n', '*' * 88, '\n')

            self.model.train()
            iter_count = 0
            train_loss = []

            if epoch < self.args.train_epochs // 2:
                self.stage = 1
            else:
                self.stage = 2
                for m in self.model.module.multi_lstms[0].parameters():
                    m.requires_grad = False

            for i, (batch_encoder_token, batch_decoder_token, batch_encoder_temporal, batch_decoder_temporal,
                    pred_token, extended_token) in enumerate(self.data_loader['shuffled_train']):
                cur_batch_size = batch_encoder_token.shape[0]
                iter_count += 1

                '''
                    batch_encoder_token: (batch_size, seq_len, feature_dim = args.enc_in)
                    batch_decoder_token: (batch_size, label_len + pred_len, feature_dim = args.dec_in)
                    batch_encoder_temporal: (batch_size, seq_len, feature_dim = args.enc_in)
                    batch_decoder_temporal: (batch_size, label_len + pred_len, feature_dim = args.dec_in)
                    pred_token: (batch_size, pred_len, feature_dim)
                '''

                outputs = self.run_one_batch(batch_encoder_token, batch_decoder_token, batch_encoder_temporal,
                                             batch_decoder_temporal)

                '''
                    outputs: (batch_size, args.pred_len, args.c_out)
                    batch_label: (batch_size, args.pred_len, 1)
                '''

                batch_label = pred_token.to(self.device)

                mse_loss = self.criterion(
                    outputs.reshape(-1),
                    batch_label.reshape(-1)
                )
                loss = mse_loss
                if self.args.use_signal_loss:
                    signal_loss = torch.mean(
                        torch.relu(
                            -1.0 * outputs.reshape(-1) * batch_label.reshape(-1)
                        )
                    )

                    loss = mse_loss + self.args.signal_loss_weight * signal_loss

                train_loss.append(loss.item())
                self.model_optimizer.zero_grad()
                loss.backward()

                if self.args.use_gradient_norm:
                    clip_value = 20
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=clip_value, norm_type=2)

                self.model_optimizer.step()

                if (i + 1) % 200 == 0:
                    print("\t iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, np.mean(train_loss)))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (train_steps - i)
                    print('\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            ###########################################################################################################
            if self.writer is None:
                self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, log_name))
            self.result_path = os.path.join(self.log_dir, log_name)

            train_loss, train_acc, train_f1 = self.validate(type='no_shuffled_train', epoch=epoch)  # 2004-2014
            val_loss, val_acc, val_f1 = self.validate(type='val', epoch=epoch)  # 2014-2017
            test_loss, test_acc, test_f1 = self.validate(type='test', epoch=epoch)  # 2017-2021
            test2_loss, test2_acc, test2_f1 = self.validate(type='test2', epoch=epoch)  # 回测 1987-2014

            if val_f1 > max_val_f1_use_val_f1:
                max_val_f1_use_val_f1 = val_f1
                max_val_acc_use_val_f1 = val_acc
                best_test_acc_use_val_f1 = test_acc
                best_test_f1_use_val_f1 = test_f1
                best_test2_acc_use_val_f1 = test2_acc
                best_test2_f1_use_val_f1 = test2_f1

            print('max val f1 use val f1', max_val_f1_use_val_f1)
            print('max val acc use val f1', max_val_acc_use_val_f1)
            print('best test acc use val f1 is', best_test_acc_use_val_f1)
            print('best test f1 use val f1 is', best_test_f1_use_val_f1)
            print('best test2 acc use val f1 is', best_test2_acc_use_val_f1)
            print('best test2 f1 use val f1 is', best_test2_f1_use_val_f1)

            self.writer.add_scalar('best_acc_use_val_f1/test', best_test_acc_use_val_f1, epoch)
            self.writer.add_scalar('best_f1_use_val_f1/test', best_test_f1_use_val_f1, epoch)
            self.writer.add_scalar('best_acc_use_val_f1/val', max_val_acc_use_val_f1, epoch)
            self.writer.add_scalar('best_f1_use_val_f1/val', max_val_f1_use_val_f1, epoch)
            self.writer.add_scalar('best_acc_use_val_f1/test2', best_test2_acc_use_val_f1, epoch)
            self.writer.add_scalar('best_f1_use_val_f1/test2', best_test2_f1_use_val_f1, epoch)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                max_val_acc_use_val_loss = val_acc
                max_val_f1_use_val_loss = val_f1
                best_test_acc_use_val_loss = test_acc
                best_test_f1_use_val_loss = test_f1
                if '1987' in self.args.data_path:
                    best_test2_acc_use_val_loss = test2_acc
                    best_test2_f1_use_val_loss = test2_f1

            print('max val acc use val loss', max_val_acc_use_val_loss)
            print('max val f1 use val loss', max_val_f1_use_val_loss)
            print('best test acc use val f1 is', best_test_acc_use_val_loss)
            print('best test f1 use val f1 is', best_test_f1_use_val_loss)
            print('best test2 acc use val loss is', best_test2_acc_use_val_loss)
            print('best test2 f1 use val loss is', best_test2_f1_use_val_loss)

            self.writer.add_scalar('best_acc_use_val_loss/test', best_test_acc_use_val_loss, epoch)
            self.writer.add_scalar('best_f1_use_val_loss/test', best_test_f1_use_val_loss, epoch)
            self.writer.add_scalar('best_acc_use_val_loss/val', max_val_acc_use_val_loss, epoch)
            self.writer.add_scalar('best_f1_use_val_loss/val', max_val_f1_use_val_loss, epoch)
            self.writer.add_scalar('best_acc_use_val_loss/test2', best_test2_acc_use_val_loss, epoch)
            self.writer.add_scalar('best_f1_use_val_loss/test2', best_test2_f1_use_val_loss, epoch)

            print("\n Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss, test_loss))

            early_stopping(val_loss, self.model, save_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(self.model_optimizer, epoch + 1, self.args)
            self.writer.add_scalar('learning rate', self.model_optimizer.param_groups[0]['lr'], epoch)

        best_model_path = save_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.writer.close()

        return self.model
