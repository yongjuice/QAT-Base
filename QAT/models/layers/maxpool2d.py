from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quantization_utils import *


class QuantizedMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=1, padding=0, arg_dict=None):
        super(QuantizedMaxPool2d, self).__init__(kernel_size, stride, padding)
        self.layer_type = 'QuantizedMaxPool2d'     
        self.runtime_helper = itemgetter('runtime_helper')(arg_dict)
        self.bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        self.kernel_size = kernel_size
        self.stride = stride
        self.maxpool = nn.MaxPool2d(self.kernel_size, self.stride, 0)

        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.padded = None

    def forward(self, x):
        if not self.padding:
            return self.maxpool(x)

        # Pad with 0
        if self.bit == 4 or self.bit == 32:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            return self.maxpool(x)

        # Pad with a single zero-point
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=self.zero_point.item())
        return self.maxpool(x)


