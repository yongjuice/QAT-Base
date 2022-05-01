from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F
from ..quantization_utils import *


class QuantizedBn2d(nn.Module):
    def __init__(self, num_features, multiplication=True, arg_dict=None):
        super(QuantizedBn2d, self).__init__()
        self.layer_type = 'QuantizedBn2d'
        self.runtime_helper = itemgetter('runtime_helper')(arg_dict)
        self.num_features = num_features
        self.w_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        t_init = 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.is_shift_neg = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.multiplication = multiplication

        self.weight = nn.Parameter(torch.zeros((1, num_features), dtype=torch.int32), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros((1, num_features), dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        out = self._subsum(x)
        if self.multiplication:
            out = self._totalsum(out)
        return out

    def _subsum(self, x):
        return self._general_subsum(x)

    def _totalsum(self, x):
        out = self._general_totalsum(x)
        return clamp_matrix(out, self.a_bit)


    def _general_subsum(self, x):
        q1q2 = x.mul(self.weight[0][None, :, None, None])
        q1z2 = x.mul(self.z2)
        q2z1 = self.weight[0].mul(self.z1)
        return q1q2 - q1z2 - q2z1[None, :, None, None] + self.z1 * self.z2 + self.bias[0][None, :, None, None]

    def _general_totalsum(self, subsum):
        if self.shift < 0:
            total = mul_and_shift(subsum << - self.shift.item(), self.M0, 0)
        else:
            total = mul_and_shift(subsum, self.M0, self.shift.item())
        return total.add(self.z3)


class FusedBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, w_bit=None, a_bit=None, arg_dict=None):
        super(FusedBnReLU, self).__init__()
        self.layer_type = 'FusedBnReLU'
        arg_w_bit, self.smooth, self.use_ste, self.runtime_helper = \
            itemgetter('bit', 'smooth', 'ste', 'runtime_helper')(arg_dict)

        w_bit = w_bit if w_bit is not None else arg_dict['bit_bn_w']
        a_bit = a_bit if a_bit is not None else arg_dict['bit']
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.activation = activation(inplace=True) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            return self._forward_impl(x)

        out = self._fake_quantized_bn(x)
        if external_range is None:
            self._update_activation_range(out)
        if self.runtime_helper.apply_fake_quantization:
            out = self._fake_quantize_activation(out, external_range)
        return out

    def _forward_impl(self, x):
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

    def _fake_quantized_bn(self, x):
        out = self.bn(x)

        with torch.no_grad():
            _x = x.detach()
            mean = _x.mean(dim=(0, 2, 3))
            var = _x.var(dim=(0, 2, 3), unbiased=False)

            weight = self.bn.weight.div(torch.sqrt(var + self.bn.eps))
            bias = self.bn.bias - weight * mean
            s, z = calc_qparams(weight.min(), weight.max(), self.w_bit)
            weight = fake_quantize(weight, s, z, self.w_bit)
            fake_out = _x * weight[None, :, None, None] + bias[None, :, None, None]

        out = STE.apply(out, fake_out)
        if self.activation:
            out = self.activation(out)
        return out

    def _update_activation_range(self, x):
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
        else:
            self.act_range[0], self.act_range[1] = get_range(x)
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

    def _fake_quantize_activation(self, x, external_range=None):
        if external_range is not None:
            s, z = calc_qparams(external_range[0], external_range[1], self.a_bit)
        else:
            s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
        return fake_quantize(x, s, z, self.a_bit, use_ste=self.use_ste)

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = s1, z1

        weight = self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        if weight.min() > 0:
            self.s2, self.z2 = calc_qparams(torch.tensor(0), weight.max(), self.w_bit)
        elif weight.max() < 0:
            self.s2, self.z2 = calc_qparams(weight.min(), torch.tensor(0), self.w_bit)
        else:
            self.s2, self.z2 = calc_qparams(weight.min(), weight.max(), self.w_bit)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)

        self.M0, self.shift = quantize_M(self.s1.type(torch.double) * self.s2.type(torch.double) / self.s3.type(torch.double))
        return self.s3, self.z3

    def get_weight_qparams(self):
        weight = self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        if weight.min() > 0:
            s, z = calc_qparams(torch.tensor(0), weight.max(), self.w_bit)
        elif weight.max() < 0:
            s, z = calc_qparams(weight.min(), torch.tensor(0), self.w_bit)
        else:
            s, z = calc_qparams(weight.min(), weight.max(), self.w_bit)
        return s, z

    def get_multiplier_qparams(self):
        return self.M0, self.shift
