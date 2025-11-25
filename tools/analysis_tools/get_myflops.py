# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')
from calflops import calculate_flops

def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('--config', default="/home/dww/OD/semi-txt/projects/MixPL/myconfig/stxt/mixpl_tood_daub.py", help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    # _forward = model.forward
    input_shape = (1,5,3,512,512)
    flops, macs, params = calculate_flops(model=model.student, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4,ignore_modules=['TOOD.neck.textatt.clip'])#
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


if __name__ == '__main__':
    main()
