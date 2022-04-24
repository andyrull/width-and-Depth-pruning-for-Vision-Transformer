# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from pruner.modules import MaskedLinear

from vit_wdpruning import VisionTransformerWithWDPruning
from tqdm import tqdm

from datasets import build_dataset
from engine import evaluate
from samplers import RASampler
import utils

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()



def count_parameters(model):
    params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'blocks' in n)
    return params


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model,test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits.view(-1, 10), y.view(-1))
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument('--data-path', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR', 'CIFAR10', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=1024, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_batch_size", default=512, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--eval', action='store_true',
                        help="Whether to eval on the eval set.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Whether to use cpu.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--distill', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument("--classifiers", type=int, nargs='+', default=[8,10],
                        help="The classifiers.")
    parser.add_argument("--classifier_choose", default=12, type=int,
                        help="Choose classifier")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Set seed
    set_seed(args)

    if args.data_set == 'CIFAR10':
        args.nb_classes = 10
    elif args.data_set == 'CIFAR100':
        args.nb_classes = 100
    else:
        args.nb_classes = 1000

    if args.arch == 'deit_small':
        embed_dim = 384
        num_heads = 6
    elif args.arch == 'deit_base':
        embed_dim = 768
        num_heads = 12
    elif args.arch == 'deit_tiny':
        embed_dim = 192
        num_heads = 3

    model = VisionTransformerWithWDPruning(num_classes=args.nb_classes,
                                         patch_size=16, embed_dim=embed_dim, depth=12, num_heads=num_heads, mlp_ratio=4,
                                         qkv_bias=True,distilled=args.distill,classifiers=args.classifiers,
                                         classifier_choose=args.classifier_choose
                                         )

    # size of unpruned model
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter of original model: \t%2.1fM" % (num_params/1000000))
    print(num_params)

    # model.load_state_dict(torch.load(args.pretrained_dir)['model'])

    model.LayerPruningAndLoadParams(dir=args.pretrained_dir)

    model.eval()

    model._make_structural_pruning()

    for n,module in model.named_modules():
        if isinstance(module, MaskedLinear):
            print(n)
            print(module.weight.shape)

    total_num_params = 0
    for name, param in model.named_parameters():
        if 'blocks' in name:
            total_num_params += (param.abs() > 1e-8).sum()


    model.to(args.device)

    # warmup!!!
    inputs = torch.ones([args.batch_size,3,224,224], dtype=torch.float).to(args.device)
    for i in range(5):
        with torch.no_grad():
            outputs = model(inputs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # do real measurement
    num_runs = 20
    torch.cuda.synchronize()
    start.record()
    for i in range(num_runs):
        with torch.no_grad():
            outputs = model(inputs)
    end.record()
    torch.cuda.synchronize()


    total_time = start.elapsed_time(end) / 1000  # s
    print('*' * 100)
    print('Num of Parameters: ', total_num_params)
    print(
        f'Remaining Parameters as compared to baseline: {(total_num_params/num_params*100):.2f}%')
    print(f"{num_runs/total_time * args.batch_size} Images / s")
    print(f"{total_time/num_runs/args.batch_size * 1000} ms / Images ")
    print('*' * 100)

    if args.eval:
        dataset_val, _ = build_dataset(is_train=False, args=args)

        if True:  # args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int( args.eval_batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        evaluate(data_loader_val, model, device)

if __name__ == "__main__":
    main()
