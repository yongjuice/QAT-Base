from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quantization_utils import *


class QActivation(nn.Module):
    """
        Activation Layer(Hardswish, Hardsigmoid, gelu ..)
    """

    def __init__(self, activation=None, arg_dict=None):
        super(QActivation, self).__init__()
        self.layer_type = 'QActivation'
        self.bit, self.smooth, self.runtime_helper, self.use_ste = itemgetter('bit', 'smooth', 'runtime_helper', 'ste')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self._activation = activation(inplace=False)

    def forward(self, x):
        x = self._activation(x)
        if not self.training:
            return x

        out = x
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                out = fake_quantize(x, s, z, self.q_max, use_ste=self.use_ste)
        else:
            self.act_range[0] = torch.min(x).item()
            self.act_range[1] = torch.max(x).item()
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)
        return out

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3
