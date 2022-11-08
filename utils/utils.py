import numpy as np
import torch


# 调整学习率函数
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'StepLR':
        lr_adjust = {
            5: 1e-4,
            60: 2e-5
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# 模型提前停止训练类定义
# 如果模型训练时，在验证集val上的loss连续patience次训练epoch都不变，那么就认为此时模型收敛，所以提前停止训练（其实此处准确讲不是loss不变，而是lss变化量小于delta）
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience    # 该参数用于控制模型训练收敛的轮数，如果连续patienc轮训练模型的loss的变化都小于delate，就停止训练
        self.verbose = verbose      # 该模型输入为True，该参数就是控制输出的，其实可以忽略
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta          # loss变化量的阈值

    def __call__(self, val_loss, model, path):  # 默认调用该函数，该函数用于判断当前训练是否需要提前结束，即“earlystopping".判断方法就是根据loss的大小变化，init函数中设置了一个阈值delta。
        score = -val_loss   # 注意此处score的定义是-val_loss
        if self.best_score is None: # 如果：best_score为空，将当前的score保存为best_score
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:  # 如果：当前训练轮的score<self.best_score + self.delta
            self.counter += 1   # 累加counter
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:          # 如果counter>=patience，就要提前停止训练了！
                self.early_stop = True
            print(f'Validation loss increased from ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):   # 保存模型
        if self.verbose:
            print()
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        torch.save(model, path + '/' + 'checkpoint_all.pth')
        print('succesfully save model in ', path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
