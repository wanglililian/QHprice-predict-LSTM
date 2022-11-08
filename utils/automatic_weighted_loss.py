import torch
import torch.nn as nn
import numpy as np

# 对于多任务的loss计算，使用该类自动对loss地每个部分进行加权，也就是在loss的各部分分别加一个权重，用来控制各部分在loss中占得比重
# 模型中可以忽略该类
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()

        params = torch.FloatTensor([1.0, 1.0])
        params.requires_grad = True
        self.params = torch.nn.Parameter(params)


    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())