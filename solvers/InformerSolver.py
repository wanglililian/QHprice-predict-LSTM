import warnings

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

from solvers.BasicSolver import BasicSolver
from models.Informer.Informer import Informer
from models.Informer.InformerStack import InformerStack


class InformerSolver(BasicSolver):
    def __init__(self, args):
        super(InformerSolver, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.frequence,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_gpu and len(self.args.gpus)>1:
            model = nn.DataParallel(model, device_ids=self.args.gpus)

        return model.to(self.device)

    def run_one_batch(self, batch_encoder_token, batch_decoder_token, batch_encoder_temporal, batch_decoder_temporal):

        # decoder input
        dec_inp = torch.zeros_like(batch_decoder_token[:, -self.args.pred_len:, :])
        dec_inp = torch.cat([batch_decoder_token[:, :self.args.label_len, :], dec_inp], dim=1).float()

        # print('encoder token = {}, encoder temporal = {}, decoder token = {}, decoder temporal = {}'.format(
        #     batch_encoder_token.shape,
        #     batch_encoder_temporal.shape,
        #     dec_inp.shape,
        #     batch_decoder_temporal.shape
        # ))

        # encoder - decoder
        outputs = self.model(
            batch_encoder_token.to(self.device),
            batch_encoder_temporal.to(self.device),
            dec_inp.to(self.device),
            batch_decoder_temporal.to(self.device)
        )

        if self.args.output_attention:
            outputs, attentions = outputs[0], outputs[1]

        return outputs.view(outputs.shape[0], self.args.pred_len, self.args.c_out)
