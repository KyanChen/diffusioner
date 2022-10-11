"""
Generate a large batch of samples from a change detection model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch as th
import torch

from guided_diffusion import logger
from guided_diffusion.script_util import (
    create_gaussian_diffusion,
    diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)
from datasets import get_loader


def main():
    args = create_argparser().parse_args()

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    logger.configure(dir=args.out_dir)

    logger.log("creating model...")
    diffusion = create_gaussian_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )

    logger.log("loading data...")
    data = load_cd_data(
        args.data_name,
        args.batch_size,
        img_size=args.img_size,
        num_workers=2,
        split=args.split,
    )

    logger.log("creating samples...")
    i = 0
    from misc.torchutils import tensor2label, save_image
    while i < args.num_samples:
        mask, model_kwargs, name = next(data)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}
        batch_size_effect = mask.shape[0]
        noise = th.randn_like(mask)
        diffusion_samples = []
        for t in range(0, args.diffusion_steps, args.diffusion_steps // 10):
            indices_np = np.zeros([batch_size_effect, ]) + t
            indices = th.from_numpy(indices_np).long().to(device)
            sample = diffusion.q_sample(
                mask,
                indices,
                noise
            )
            diffusion_samples.append(sample)
        diffusion_samples = torch.concat(diffusion_samples, dim=0)
        img = tensor2label(diffusion_samples, min_max=(0, 1))
        out_path = os.path.join(args.out_dir, name[0])
        save_image(img, out_path)

        logger.log(f'save result {out_path}...')
        i += batch_size_effect

    logger.log("sampling complete")


def load_cd_data(data_name, batch_size, img_size, class_cond=False, num_workers=2,
                 split='test'):

    data = get_loader(is_train=False, batch_size=batch_size,
                      num_workers=num_workers, img_size=img_size,
                      dataset_type='CDDataset',
                      data_name=data_name, norm=True,
                      split=split)
    # while True:
    for batch in data:
        A, B, mask =batch['A'], batch['B'], batch['mask']
        img_concat = torch.concat([A, B], dim=1)
        model_kwargs = dict(bi_img=img_concat)
        yield mask, model_kwargs, batch['name']


def create_argparser():
    defaults = dict()
    defaults.update(diffusion_defaults())
    defaults_ = dict(
        data_name="LEVIR",
        split='test1',
        out_dir='../out_dir/levir_forward_sample',
        img_size=256,
        num_samples=32,
        batch_size=1,
        diffusion_steps=1000,
        # timestep_respacing='500',
        use_ddim=False,
    )
    defaults.update(defaults_)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
