import argparse
import os
from utils.torch_dataset import get_data_loaders

parser = argparse.ArgumentParser(description='[PyTorch] Quantization')
parser.add_argument('--worker', default=4, type=int, help='Number of workers for input data loader')
parser.add_argument('--mode', default='fine', type=str, help="pre/fine/eval/lip")
parser.add_argument('--imagenet', default='', type=str, help="ImageNet dataset path")
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--val_batch', default=0, type=int, help='Validation batch size')

parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")
parser.add_argument('--horovod', default=False, type=bool, help="Use distributed training with horovod")
args, _ = parser.parse_known_args()

if args.imagenet:
    args.dataset = 'imagenet'
if args.dataset == 'cifar':
    args.dataset = 'cifar10'
if not args.val_batch:
    args.val_batch = 256 if args.dataset != 'imagenet' else 128

if __name__ == '__main__':
    data_loaders = get_data_loaders(args)

    from QAT.qat import main
    main(args, data_loaders)