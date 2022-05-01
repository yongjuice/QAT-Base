from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from ..quantization_utils import *
from .activation import *


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, activation=None, multiplication=True, arg_dict=None):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantizedLinear'
        bit, self.symmetric, self.runtime_helper = itemgetter('bit', 'symmetric', 'runtime_helper')(arg_dict)
        self.w_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.out_features = out_features

        self.is_bias = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.quantized_bias = nn.Parameter(torch.zeros((1, out_features), dtype=torch.int32), requires_grad=False)

        self.sum_a2 = nn.Parameter(torch.zeros((1, out_features), dtype=torch.int32), requires_grad=False)
        self.multiplication = multiplication

        t_init = 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.is_shift_neg = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)    #
        self.hardswish_6 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.hardswish_3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight, None)
        out = self._subsum(x, out.type(torch.cuda.LongTensor))
        if self.multiplication:
            out = self._totalsum(out)
        return out

    def _subsum(self, x, y):
        return self._general_subsum(x, y)

    def _totalsum(self, x):
        out = self._general_totalsum(x)
        return clamp_matrix(out, self.a_bit)

    def _general_subsum(self, x, sum_q1q2):
        if self.is_bias:
            sum_q1q2.add_(self.quantized_bias[0][None, :])

        if not self.symmetric:
            sum_a1 = torch.sum(x, dim=1).mul(self.z2)
            sum_a2 = self.sum_a2.mul(self.z1)

            nz1z2 = x.size(1) * self.z1 * self.z2
            subsum = sum_q1q2.add(nz1z2)
            subsum = subsum.sub(sum_a1[:, None])
            subsum = subsum.sub(sum_a2)
        else:
            subsum = sum_q1q2.sub(self.sum_a2.mul(self.z1))
        return subsum

    def _general_totalsum(self, subsum):
        if self.shift < 0:
            total = mul_and_shift(subsum << - self.shift.item(), self.M0, 0)
        else:
            total = mul_and_shift(subsum, self.M0, self.shift.item())
        return total.add(self.z3)


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, is_classifier=False,
                 w_bit=None, a_bit=None, arg_dict=None):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'

        self.arg_dict = arg_dict
        bit, self.symmetric, self.smooth, self.use_ste, self.runtime_helper = itemgetter('bit', 'symmetric', 'smooth', 'ste', 'runtime_helper')(arg_dict)

        w_bit = w_bit if w_bit is not None else bit
        a_bit = a_bit if a_bit is not None else bit
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        if not self.training:
            x = self.fc(x)
            if self._activation:
                x = self._activation(x)
            return x

        w = self.fc.weight.detach()
        s, z = calc_qparams(w.min(), w.max(), self.w_bit, symmetric=self.symmetric)
        w = fake_quantize(self.fc.weight, s, z, self.w_bit, symmetric=self.symmetric, use_ste=self.use_ste)

        out = F.linear(x, w, self.fc.bias)
        if self._activation:
            out = self._activation(out)

        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
                out = fake_quantize(out, s, z, self.a_bit, use_ste=self.use_ste)
        else:
            self.act_range[0], self.act_range[1] = get_range(out)
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)
        return out

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = s1, z1
        self.s2, self.z2 = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.w_bit,
                                        symmetric=self.symmetric)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)

        self.M0, self.shift = quantize_M(self.s1.type(torch.double) * self.s2.type(torch.double) / self.s3.type(torch.double))
        return self.s3, self.z3