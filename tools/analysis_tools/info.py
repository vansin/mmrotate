# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmengine import Config
from functools import partial

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules
from mmengine.runner import Runner
from torchview import draw_graph

register_all_modules()

from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape, please use --shape h w')

    input_shape = (1, 3, h, w)
    cfg = Config.fromfile(args.config)

    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    for idx, data_batch in enumerate(dataloader):
        print(idx, data_batch)
        break

    model = MODELS.build(cfg.model)
    _forward = model.forward

    data = model.data_preprocessor(data_batch)
    model.forward = partial(_forward, data_samples=data['data_samples'])


    summary(model, data['inputs'].shape, depth=3)
    # summary(model, (1, 3, 1024, 1024), depth=3)
    model_graph = draw_graph(model, input_size=data['inputs'].shape)
    model_graph.visual_graph.render('1', view=False)

if __name__ == '__main__':
    main()