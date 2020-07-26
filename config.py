#  -*- coding:utf-8 -*-
#  Editor      : Pycharm
#  File        : config.py
#  Created     : 2020/7/22 下午11:56
#  Description : input params script for xunfei


import argparse


def get_cfgs():
    parser = argparse.ArgumentParser(description='Hyperparams')

    # data
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="whether use gpu, 0 dont use"
    )
    parser.add_argument(
        '--train_annotation_path', type=str,
        default="/home/nofalling/Downloads/data/eval/validation.json",
        help='train_ann_path')
    parser.add_argument(
        '--eval_annotation_path', type=str,
        default="/home/nofalling/Downloads/data/eval/validation.json",
        help="eval_ann_path")
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./saved_models'
    )

    # train stage
    parser.add_argument(
        '--batch_size', nargs='?', type=int, default=16, help='Batch Size')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--optimizer', type=str, default='sgd', help="['sgd', 'adam']"
    )
    parser.add_argument(
        '--lr', nargs='?', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4,
        help='weight decay of trainable params')
    parser.add_argument(
        '--pretrain', type=int, default=0,
        help='Path to previous saved model to restart from')
    parser.add_argument(
        '--n_epoch', nargs='?', type=int, default=600, help='# of the epochs')
    parser.add_argument(
        '--scheduler', type=str, default='multistep_lr'
    )

    # network hyper params
    parser.add_argument(
        "--kernel_num", type=int, default=7)
    parser.add_argument(
        "--min_scale", type=float, default=0.4)

    # network info
    parser.add_argument(
        '--img_size', nargs='?', type=int, default=640, help='Height of the input image')
    parser.add_argument(
        '--arch', nargs='?', type=str, default='resnet50')


    # parser.add_argument('--schedule', type=int, nargs='+', default=[200, 400],
    #                     help='Decrease learning rate at these epochs.')
    #
    #
    # parser.add_argument('--resume', nargs='?', type=str, default=None,
    #                     help='Path to previous saved model to restart from')

    # parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
    #                     help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    return args