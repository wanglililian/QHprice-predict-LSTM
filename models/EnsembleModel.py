import torch.nn as nn
import torch

class EnsembleModel(nn.Module):
    def __init__(self, model_type_list=[], model_path_list=[], seq_len_list=[], weight_list=[0.9,0.1]):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleDict()
        self.seq_len_list = seq_len_list
        params = torch.FloatTensor(weight_list)
        params.requires_grad = True
        self.params = torch.nn.Parameter(params)

        for i, (model_type, model_path) in enumerate(zip(model_type_list, model_path_list)):
            print(model_path)
            model = torch.load(model_path).module
            self.models[model_type] = model

    def forward(self, x):
        outputs = None
        for i, (model_type, model) in enumerate(self.models.items()):
            if outputs is None:
                outputs = self.params[i] * self.models[model_type](x[:,-self.seq_len_list[i]:,:])
            else:
                outputs = outputs + self.params[i] * self.models[model_type](x[:,-self.seq_len_list[i]:,:])

        return outputs


