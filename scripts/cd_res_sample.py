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

from guided_diffusion import logger
from guided_diffusion.script_util import (
    cd_model_and_diffusion_defaults,
    cd_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from datasets import get_loader
import torch


def main():
    args = create_argparser().parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger.configure(dir=args.out_dir)

    logger.log("creating model...")
    model, diffusion = cd_create_model_and_diffusion(
        **args_to_dict(args, cd_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_cd_data(
        args.data_name,
        args.batch_size,
        img_size=args.img_size,
        num_workers=2,
        split=args.split,
    )


    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        mask, model_kwargs, name = next(data)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}
        batch_size_effect = mask.shape[0]
        with torch.no_grad():
            if args.use_ddim:
                sample = diffusion.ddim_sample_loop(
                    model,
                    (batch_size_effect, 1, args.img_size, args.img_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=True,
                )
            else:
                sample = diffusion.p_sample_loop(
                    model,
                    (batch_size_effect, 1, args.img_size, args.img_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=True,
                )

        out_color = False
        if not out_color:
            sample = (sample > 0.5).to(th.uint8)
        sample = sample[:, 0]
        all_images.append(sample)
        from misc.torchutils import save_image, norm
        from misc.imutils import apply_colormap
        for i, item in enumerate(sample):
            out_path = os.path.join(args.out_dir, name[i])
            out = item.cpu().numpy()
            if out_color:
                out = (out - out.min()) / (out.max()-out.min() + 10e-5)
            out = (out * 255).astype(np.uint8)
            save_image(out, out_path)
            print(f'save result {out_path}...')
        # logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr)

    logger.log("sampling complete")


def load_cd_data(data_name, batch_size, img_size, class_cond=False, num_workers=2,
                 split='test'):

    data = get_loader(is_train=False, batch_size=batch_size,
                      num_workers=num_workers, img_size=img_size,
                      dataset_type='CDDataset',
                      data_name=data_name, norm=True,
                      split=split)
    # while True:
    for i, batch in enumerate(data):
        # if i<20:
        #    continue
        A, B, mask =batch['A'], batch['B'], batch['mask']
        img_concat = torch.concat([A, B], dim=1)
        model_kwargs = dict(bi_img=img_concat)
        yield mask, model_kwargs, batch['name']


def create_argparser():
    defaults = dict()
    defaults.update(cd_model_and_diffusion_defaults())
    defaults_ = dict(
        data_name="LEVIR", #LEVIR  WHU
        model_name='V4',
        split='test',
        out_dir='../out_dir/zht_levir_train_V4_e60_b8/test_e60_step50',
        learn_sigma=False,
        img_size=256,
        clip_denoised=True,
        num_samples=2048,
        batch_size=16,
        diffusion_steps=1000,
        timestep_respacing='50',  # ddim25
        use_ddim=False,
        base_samples="",
        model_path="../out_dir/zht_levir_train_V4_e60_b8/modellatest.pt",
    )
    defaults.update(defaults_)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
