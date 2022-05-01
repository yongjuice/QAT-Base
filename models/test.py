from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F
import time

from quantization_utils import *

# torch.Size([256, 3, 32, 32]) torch.Size([16, 3, 3, 3])

x = torch.rand(256,3,32,32).cuda()
weight = torch.rand(16,3,3,3).cuda()
bias = torch.rand(16).cuda()
# x = torch.randn(2,3,5,5).cuda()
# weight = torch.rand(2,3,3,3).cuda()
# bias = torch.rand(2).cuda()

stride = 1
# stride=2

s1, z1 = calc_qparams(torch.min(x), torch.max(x), 15)
s2, z2 = calc_qparams(torch.min(weight), torch.max(weight), 15)

input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
filter_batch, filter_ch, filter_col, filter_row = weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]

sum_q1q2 = F.conv2d(x, weight, None, 1, (0, 0))
print(sum_q1q2.shape)
exit()

for output_ch in range(filter_batch):
    sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add(bias[output_ch])

output_col = sum_q1q2.shape[2]
output_row = sum_q1q2.shape[3]
sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
test_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()
test_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()

for output_ch in range(filter_batch):
    sum_a2[output_ch] = torch.sum(weight.data[output_ch, :, :, :]).mul(z1)
test_a2 = torch.sum(weight.data).mul(z1)


origin_start = time.time()
for o_col in range(output_col):
    for o_row in range(output_row):
        col_st, col_end = o_col * stride, o_col * stride + filter_col
        row_st, row_end = o_row * stride, o_row * stride + filter_row
        sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(z2)

# for o_col in range(output_col):
#     for o_row in range(output_row):
#         col_st, col_end = o_col * stride, o_col * stride + filter_col
#         row_st, row_end = o_row * stride, o_row * stride + filter_row
#         test_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3))
# test_a1 = test_a1.mul(z2)
# mine_end = time.time()

nz1z2 = input_ch * filter_col * filter_row * z1 * z2
sum_q1q2 = sum_q1q2.add(nz1z2)

# test_q1q2 = sum_q1q2

# origin_start = time.time()
for i_batch in range(input_batch):
    sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])
# origin_end = time.time()
# print("origin time: ", origin_end-origin_start)
# mine_start = time.time()
# test_q1q2 = torch.sub(test_q1q2, sum_q1q2)
# mine_end = time.time()
# print("mine time : ", mine_end - mine_start)
#
# print("Faster ", (origin_end-origin_start) - (mine_end-mine_start))
# print("Faster ", (mine_end-mine_start) / (origin_end-origin_start)*100, "%")
#
# if torch.equal(sum_a1, test_a1):
#     print(True)
# exit()

test_q1q2 = sum_q1q2
origin_start = time.time()
for out_c in range(filter_batch):
    sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[out_c])
origin_end = time.time()
print("origin time: ", origin_end-origin_start)
mine_start = time.time()
test_q1q2 = torch.sub(test_q1q2, test_a2)
mine_end = time.time()
print("mine time : ", mine_end - mine_start)

print("Faster ", (origin_end-origin_start) - (mine_end-mine_start))
print("Faster ", (mine_end-mine_start) / (origin_end-origin_start)*100, "%")

if torch.equal(sum_q1q2, test_q1q2):
    print("True")
else:
    print(sum_q1q2)
    print(test_q1q2)
exit()

# if shift < 0:
#     multiplied = multiply_M((sum_q1q2.type(torch.cuda.LongTensor) << - shift.item()), M0)
#     total = shifting(multiplied, 0)
# else:
#     multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
#     total = shifting(multiplied, shift.item())


# total = total.add(z3)

total = torch.clamp(total, 0, 15)

print("RESULT : ", total.type(torch.cuda.FloatTensor))