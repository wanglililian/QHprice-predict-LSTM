import warnings

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver, EarlyStopping, adjust_learning_rate
from models.CEEMDANLSTM import CEEMDANLSTM

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


class CEEMDANLSTMSolver(BasicSolver):
    def __init__(self, args):
        super(CEEMDANLSTMSolver, self).__init__(args)

    def _build_model(self):
        args = self.args
        model = CEEMDANLSTM(
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

        return model.to(self.device)

    def prepare_dataset(self):
        args = self.args

        main_df = pd.read_csv(
            os.path.join(args.root_path, 'Latest_CL_Data/Day_Week_Month/WTI_1987-2021_NoMinMax_CEEMDAN.csv'),
            header=0,
            index_col='Date'
        )

        self.dataset['test2'] = self._get_dataset(main_df['1987':'2004'], natural_week=True)
        self.dataset['shuffled_train'] = self._get_dataset(main_df['2004':'2014'], natural_week=False)
        self.dataset['no_shuffled_train'] = self._get_dataset(main_df['2004':'2014'], natural_week=True)
        self.dataset['val'] = self._get_dataset(main_df['2014':'2017'], natural_week=True)
        self.dataset['test'] = self._get_dataset(main_df['2017':], natural_week=True)

        # use multi data
        # for path in [
        #     'Latest_CL_Data/Day_Week_Month/B00_IPE_1988.06.23-2021.04.30.csv',
        #     # 'Latest_CL_Data/Day_Week_Month/DJUSEN_GI_1992.1.2—2021.4.30.csv',
        #     'Latest_CL_Data/Day_Week_Month/DJI_GI_1988.6.23-2021.4.30.csv',
        #     'Latest_CL_Data/Day_Week_Month/000001.SH_1990.12.19-2021.4.30.csv',
        # ]:
        #     extended_df = pd.read_csv(
        #         os.path.join(args.root_path, path),
        #         header=0,
        #         index_col='Date'
        #     )
        #
        #     self.dataset['shuffled_train'] += self._get_dataset(extended_df['2004':'2014'], natural_week=False)
        #     self.dataset['no_shuffled_train'] += self._get_dataset(extended_df['2004':'2014'], natural_week=True)

        self.data_loader['shuffled_train'] = self._get_dataloader(self.dataset['shuffled_train'], shuffle=True)
        self.data_loader['no_shuffled_train'] = self._get_dataloader(self.dataset['no_shuffled_train'], shuffle=False)
        self.data_loader['val'] = self._get_dataloader(self.dataset['val'], shuffle=False)
        self.data_loader['test'] = self._get_dataloader(self.dataset['test'], shuffle=False)
        self.data_loader['test2'] = self._get_dataloader(self.dataset['test2'], shuffle=False)

    def run_one_batch(self, batch_encoder_token, batch_decoder_token, batch_x_temporal, batch_y_temporal):
        '''
        Input:
            batch_encoder_token: (batch_size, seq_len, args.enc_in * 3)
        Output:
            outputs:  (batch_size, pred_len, args.c_out * 4)
            labels:  (batch_size, pred_len, args.c_out * 4)
        '''
        outputs = self.model(
            batch_encoder_token.to(self.device),
        )

        return outputs.view(outputs.shape[0], self.args.pred_len, self.args.c_out * 4)

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

            for i, (batch_encoder_token, batch_decoder_token, batch_encoder_temporal, batch_decoder_temporal,
                    pred_token, extended_token) in enumerate(
                self.data_loader['shuffled_train']):
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

                if i% 20 == 0:
                    print(batch_encoder_token.shape, outputs.shape, batch_label.shape)
                    print(batch_encoder_token[0,-1,:])
                    print(outputs[0, 0, :])
                    print(batch_label[0, 0, :])
                    print()

                if self.args.task == 'Regression':
                    mse_loss = self.criterion(
                        outputs[:,:,1:],
                        batch_label[:,:,1:]
                    )
                    mse_loss1 = self.criterion(outputs[:,:,1],batch_label[:,:,1])
                    mse_loss2 = self.criterion(outputs[:, :, 2], batch_label[:, :, 2])
                    mse_loss3 = self.criterion(outputs[:, :, 3], batch_label[:, :, 3])

                    loss = mse_loss1 + mse_loss2  + mse_loss3

                    if self.args.use_signal_loss:
                        signal_loss = torch.mean(
                            torch.relu(
                                -1.0 * outputs[:,:,0].reshape(-1) * batch_label[:,:,0].reshape(-1)
                            )
                        )

                        # signal_loss = torch.mean(
                        #     torch.relu(
                        #         -1.0 * outputs.reshape(-1) * batch_label.reshape(-1)
                        #     )
                        # )

                        # loss = self.awl(mse_loss, signal_loss)

                        loss = mse_loss + signal_loss

                        # print('mse loss = {}, signal loss = {}'.format(mse_loss, signal_loss))

                elif self.args.task == 'Classification':
                    loss = self.criterion(
                        outputs.reshape(-1, self.args.c_out),
                        batch_label.reshape(-1)
                    )

                train_loss.append(loss.item())
                self.model_optimizer.zero_grad()
                loss.backward()

                if self.args.use_gradient_norm:
                    # grad_norm = _get_grad_norm(self.model)
                    # self.history_grad_norm.append(grad_norm)
                    # clip_value = np.percentile(self.history_grad_norm, 25)
                    clip_value = 20
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=clip_value, norm_type=2)
                    # print('use gradient norm, clip value = {}'.format(clip_value))

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

            train_loss, train_acc, train_f1 = self.validate(type='no_shuffled_train', epoch=epoch)
            val_loss, val_acc, val_f1 = self.validate(type='val', epoch=epoch)
            test_loss, test_acc, test_f1 = self.validate(type='test', epoch=epoch)

            if '1987' in self.args.data_path:
                test2_loss, test2_acc, test2_f1 = self.validate(type='test2', epoch=epoch)

            if self.args.use_signal_loss:
                self.writer.add_scalar('automatic weighted loss/sigma1', self.awl.params[0].item(), epoch)
                self.writer.add_scalar('automatic weighted loss/sigma2', self.awl.params[1].item(), epoch)

            if val_f1 > max_val_f1_use_val_f1:
                max_val_f1_use_val_f1 = val_f1
                max_val_acc_use_val_f1 = val_acc
                best_test_acc_use_val_f1 = test_acc
                best_test_f1_use_val_f1 = test_f1
                if '1987' in self.args.data_path:
                    best_test2_acc_use_val_f1 = test2_acc
                    best_test2_f1_use_val_f1 = test2_f1

            print('max val f1 use val f1', max_val_f1_use_val_f1)
            print('max val acc use val f1', max_val_acc_use_val_f1)
            print('best test acc use val f1 is', best_test_acc_use_val_f1)
            print('best test f1 use val f1 is', best_test_f1_use_val_f1)

            self.writer.add_scalar('best_acc_use_val_f1/test', best_test_acc_use_val_f1, epoch)
            self.writer.add_scalar('best_f1_use_val_f1/test', best_test_f1_use_val_f1, epoch)
            self.writer.add_scalar('best_acc_use_val_f1/val', max_val_acc_use_val_f1, epoch)
            self.writer.add_scalar('best_f1_use_val_f1/val', max_val_f1_use_val_f1, epoch)

            if '1987' in self.args.data_path:
                print('best test2 acc use val f1 is', best_test2_acc_use_val_f1)
                print('best test2 f1 use val f1 is', best_test2_f1_use_val_f1)
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

            self.writer.add_scalar('best_acc_use_val_loss/test', best_test_acc_use_val_loss, epoch)
            self.writer.add_scalar('best_f1_use_val_loss/test', best_test_f1_use_val_loss, epoch)
            self.writer.add_scalar('best_acc_use_val_loss/val', max_val_acc_use_val_loss, epoch)
            self.writer.add_scalar('best_f1_use_val_loss/val', max_val_f1_use_val_loss, epoch)

            if '1987' in self.args.data_path:
                print('best test2 acc use val loss is', best_test2_acc_use_val_loss)
                print('best test2 f1 use val loss is', best_test2_f1_use_val_loss)
                self.writer.add_scalar('best_acc_use_val_loss/test2', best_test2_acc_use_val_loss, epoch)
                self.writer.add_scalar('best_f1_use_val_loss/test2', best_test2_f1_use_val_loss, epoch)

            print("\n Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss, test_loss))

            early_stopping(val_loss, self.model, save_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.model_optimizer, epoch + 1, self.args)
            self.writer.add_scalar('learning rate', self.model_optimizer.param_groups[0]['lr'], epoch)

        best_model_path = save_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.writer.close()
        return self.model

    def add_metrics(self, cur_prices, preds, trues, epoch, type):
        '''
        input:
            cur_prices: (batch_num, 1, feature_num = args.enc_in)
            preds:      (batch_num, pred_len, feature_num = args.c_out)
            trues:      (batch_num, pred_len, feature_num = args.c_out)
        '''

        args = self.args
        print('\n', type)

        print('preds.shape = {}, trues.shape = {}'.format(preds.shape, trues.shape))
        # use only the first prediction
        preds = preds[:, 0, 0].reshape(-1)
        trues = trues[:, 0, 0].reshape(-1)
        last_prices = cur_prices[:, 0, 0].reshape(-1)

        pred_prices = preds
        true_prices = trues

        pred_returns = (pred_prices - last_prices) / last_prices
        true_returns = (true_prices - last_prices) / last_prices

        mape_of_price = MAPE(pred_prices, true_prices)

        print('MAPE_of_Price = {}'.format(mape_of_price))
        self.writer.add_scalar('MAPE_OF_PRICE/{}'.format(type), mape_of_price, epoch)

        figs = plot_results(pred_prices, true_prices)

        # preds is Returns of Close Price
        preds = [1 if i >= 0 else 0 for i in pred_returns]
        trues = [1 if i >= 0 else 0 for i in true_returns]

        result = classification_report(y_true=trues, y_pred=preds)
        print(result)
        result = classification_report(y_true=trues, y_pred=preds, output_dict=True)
        print('MA_ACC = {}'.format(result['accuracy']))

        self.writer.add_scalar('MA_ACC/{}'.format(type), result['accuracy'], epoch)
        self.writer.add_scalar('F1-Score/{}'.format(type), result['macro avg']['f1-score'], epoch)

        for i, fig in enumerate(figs):
            self.writer.add_figure(tag='{}/fig{}'.format(type, i), figure=fig, global_step=epoch)

        return result['accuracy'], result['macro avg']['f1-score']
