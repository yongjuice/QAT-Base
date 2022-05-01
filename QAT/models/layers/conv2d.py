from operator import itemgetter

import torch.nn.functional as F

from ..quantization_utils import *


# import int_quantization


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None,
                 dilation=1, groups=1, bias=False, is_first=False, multiplication=True, arg_dict=None):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias)
        self.layer_type = 'QuantizedConv2d'
        bit, self.per_channel, self.symmetric, self.runtime_helper, self.default_batch = \
            itemgetter('bit', 'per_channel', 'symmetric', 'runtime_helper', 'val_batch')(arg_dict)
        self.w_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.is_bias = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.quantized_bias = nn.Parameter(torch.zeros((1, out_channels), dtype=torch.int32),
                                           requires_grad=False)
        self.sum_a2 = nn.Parameter(torch.zeros((1, out_channels, 1, 1), dtype=torch.int32), requires_grad=False)
        self.sum_a1 = None  # for faster inference      ###

        self.out_channels = out_channels
        self.multiplication = multiplication

        self.s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(0,  dtype=torch.int32), requires_grad=False)

        self.is_shift_neg = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        if self.per_channel:
            self.s2 = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32), requires_grad=False)
            self.z2 = nn.Parameter(torch.zeros(out_channels, dtype=torch.int32), requires_grad=False)
            self.M0 = nn.Parameter(torch.zeros((1, out_channels), dtype=torch.int32),
                                   requires_grad=False)
            self.shift = nn.Parameter(torch.zeros((1, out_channels), dtype=torch.int32),
                                      requires_grad=False)
        else:
            self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
            self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
            self.M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
            self.shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        x, out = self._conv_impl(x)
        out = self._subsum(x, out)
        if self.multiplication:
            out = self._totalsum(out)
        return out

    def _conv_impl(self, x):
        padded = x

        # Pad if needed
        if self.padding[0] > 0:
            to_pad = (self.padding[0], self.padding[0], self.padding[1], self.padding[1])

            padded = F.pad(x, to_pad, mode='constant', value=self.z1.item())

        out = F.conv2d(padded, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
        return padded.type(torch.cuda.IntTensor), out.type(torch.cuda.LongTensor)

    def _subsum(self, x, y):
        return self._general_subsum(x, y)

    def _totalsum(self, x):
        out = self._general_totalsum(x)
        return clamp_matrix(out, self.a_bit)

    def _general_subsum(self, x, sum_q1q2):
        if self.is_bias:
            sum_q1q2 = sum_q1q2.add(self.quantized_bias[0][None, :, None, None])

        if not self.symmetric:
            input_batch, input_ch = x.shape[0], x.shape[1]
            filter_col, filter_row = self.weight.shape[2], self.weight.shape[3]
            stride = self.stride[0]
            output_col, output_row = sum_q1q2.shape[2], sum_q1q2.shape[3]
            if self.sum_a1 is None or self.sum_a1.shape[0] != input_batch:
                self.sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32, device='cuda')
            for o_col in range(output_col):
                for o_row in range(output_row):
                    col_st, col_end = o_col * stride, o_col * stride + filter_col
                    row_st, row_end = o_row * stride, o_row * stride + filter_row
                    self.sum_a1[:x.size(0), o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end],
                                                                      (1, 2, 3))
            sum_a1 = self.sum_a1[:x.size(0)].mul(self.z2)

            sum_a2 = self.sum_a2.mul(self.z1)
            nz1z2 = input_ch * filter_col * filter_row * self.z1 * self.z2
            subsum = sum_q1q2.add(nz1z2)
            subsum = torch.sub(subsum, sum_a1[:, None, :, :])
            subsum = torch.sub(subsum, sum_a2)
        else:
            subsum = sum_q1q2.sub(self.sum_a2.mul(self.z1))
        return subsum

    def _general_totalsum(self, subsum):
        # if self.shift < 0:
        #     total = mul_and_shift(subsum << - self.shift.item(), self.M0, 0)
        # else:
        #     total = mul_and_shift(subsum, self.M0, self.shift.item())

        if self.per_channel:
            M0 = self.M0[:, :, None, None]
            shift = self.shift[:, :, None, None]
        else:
            M0 = self.M0
            shift = self.shift

        mask = self.runtime_helper.mask_4d[:subsum.size(0)]
        if not self.is_shift_neg:
            total = mul_and_shift(subsum, M0, shift, mask)
        else:
            zero = self.runtime_helper.izero
            neg_shift = torch.where(shift < zero, - shift, zero)
            shift = torch.where(shift >= zero, shift, zero)
            subsum = subsum << neg_shift
            total = mul_and_shift(subsum, M0, shift, mask)
        return total.add(self.z3)


class FusedConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=None, activation=None, w_bit=None, a_bit=None, arg_dict=None):
        super(FusedConv2d, self).__init__()
        self.layer_type = 'FusedConv2d'
        self.groups = groups

        self.arg_dict = arg_dict
        self.per_channel, self.symmetric, self.smooth, self.fold_convbn, self.use_ste, self.runtime_helper = itemgetter('per_channel', 'symmetric', 'smooth', 'fold_convbn', 'ste', 'runtime_helper')(arg_dict)

        w_bit = w_bit if w_bit is not None else arg_dict['bit']
        a_bit = a_bit if a_bit is not None else arg_dict['bit']
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=self.groups, bias=bias, dilation=dilation)
        self._norm_layer = norm_layer(out_channels) if norm_layer else None
        self._activation = activation(inplace=False) if activation else None
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, external_range=None):

        # int_quantization.float2gemmlowp(x, 1.0, 1.0, 1, False, False, x)
        if not self.training:
            x = self.conv(x)
            if self._norm_layer:
                x = self._norm_layer(x)
            if self._activation:
                x = self._activation(x)
            return x

        if self.fold_convbn:
            return self._norm_folded(x, external_range)
        else:
            return self._general(x, external_range)

    def _general(self, x, external_range=None):
        zero = self.runtime_helper.fzero
        if self.per_channel:
            w = fake_quantize_per_output_channel(self.conv.weight, self.w_bit, zero,
                                                 symmetric=self.symmetric, use_ste=self.use_ste)
        else:
            w = self.conv.weight.detach()
            s, z = calc_qparams(w.min(), w.max(), self.w_bit, symmetric=self.symmetric, zero=zero)
            w = fake_quantize(self.conv.weight, s, z, self.w_bit,
                              symmetric=self.symmetric, use_ste=self.use_ste)

        out = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        if self._activation:
            out = self._activation(out)

        if external_range is not None:
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(external_range[0], external_range[1], self.a_bit)
                out = fake_quantize(out, s, z, self.a_bit, use_ste=self.use_ste)
        else:
            if self.apply_ema:
                self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
                    out = fake_quantize(out, s, z, self.a_bit, use_ste=self.use_ste)
            else:
                self.act_range[0], self.act_range[1] = get_range(out)
                self.apply_ema.data = torch.tensor(True, dtype=torch.bool)
        return out

    def _norm_folded(self, x, external_range=None):
        general_out = self.conv(x)
        general_out = self._norm_layer(general_out)
        if self._activation:
            general_out = self._activation(general_out)

        with torch.no_grad():
            alpha, beta, mean, var, eps = self._norm_layer.weight, self._norm_layer.bias, self._norm_layer.running_mean, \
                                          self._norm_layer.running_var, self._norm_layer.eps
            n_channel = self.conv.weight.shape[0]

            folded_weight = self.conv.weight.clone().detach()
            folded_bias = beta.clone().detach()
            for c in range(n_channel):
                folded_weight.data[c] = folded_weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
                folded_bias.data[c] = folded_bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))

            s, z = calc_qparams(torch.min(folded_weight), torch.max(folded_weight), self.w_bit,
                                symmetric=self.symmetric)
            fq_folded_weight = fake_quantize(folded_weight, s, z, self.w_bit, use_ste=False)

            folded_out = F.conv2d(x, fq_folded_weight, folded_bias, self.conv.stride, self.conv.padding,
                                  self.conv.dilation, self.conv.groups)
            if self._activation:
                folded_out = self._activation(folded_out)

            if external_range is None:
                if self.apply_ema:
                    self.act_range[0], self.act_range[1] = ema(folded_out, self.act_range, self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
                        folded_out = fake_quantize(folded_out, s, z, self.a_bit, use_ste=False)
                else:
                    self.act_range[0] = torch.min(folded_out).item()
                    self.act_range[1] = torch.max(folded_out).item()
                    self.apply_ema.data = torch.tensor(True, dtype=torch.bool)
            else:
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(external_range[0], external_range[1], self.a_bit)
                    folded_out = fake_quantize(folded_out, s, z, self.a_bit, use_ste=False)
        return STE.apply(general_out, folded_out)

    def fold_conv_and_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, 'Do not fuse layers while training.'
        alpha, beta, mean, var, eps = self._norm_layer.weight, self._norm_layer.bias, self._norm_layer.running_mean, \
                                      self._norm_layer.running_var, self._norm_layer.eps
        n_channel = self.conv.weight.shape[0]
        self.conv.bias = nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))
        self._norm_layer = nn.Identity()

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = s1, z1

        zero = self.runtime_helper.fzero
        if self.per_channel:
            self.s2, self.z2 = calc_qparams_per_output_channel(self.conv.weight, self.w_bit,
                                                               symmetric=self.symmetric, zero=zero)
        else:
            self.s2, self.z2 = calc_qparams(self.conv.weight.min(), self.conv.weight.max(), self.w_bit,
                                            symmetric=self.symmetric, zero=zero)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)

        if self.per_channel:
            self.M0 = torch.zeros((1, self.out_channels), dtype=torch.int32)
            self.shift = torch.zeros((1, self.out_channels), dtype=torch.int32)
            m_per_channel = self.s1.type(torch.double) * self.s2.type(torch.double) / self.s3.type(torch.double)
            for channel in range(self.out_channels):
                self.M0[0][channel], self.shift[0][channel] = quantize_M(m_per_channel[channel])
        else:
            self.M0, self.shift = quantize_M(
                self.s1.type(torch.double) * self.s2.type(torch.double) / self.s3.type(torch.double))
        return self.s3, self.z3