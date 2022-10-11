"""
Train a change detection model.
"""
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#os.path.dirname作用为返回上级目录

sys.path.append(BASE_DIR)
import argparse

import torch.nn.functional as F
import torch
from guided_diffusion import logger
from datasets import get_loader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    cd_model_and_diffusion_defaults,
    cd_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    out_dir = args.out_dir
    logger.configure(dir=out_dir)

    logger.log("creating model...")
    model, diffusion = cd_create_model_and_diffusion(
        **args_to_dict(args, cd_model_and_diffusion_defaults().keys())
    )
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_cd_data(
        args.data_name,
        args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        split=args.split,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_cd_data(data_name, batch_size, img_size,
                 num_workers=4,
                 split='train'):

    data = get_loader(is_train=True, batch_size=batch_size,
                      num_workers=num_workers, img_size=img_size,
                      dataset_type='CDDataset',
                      data_name=data_name, norm=False,
                      split=split)
    from datasets.transforms import get_cd_augs
    cd_augs = get_cd_augs(imgz_size=img_size)
    while True:
        for batch in data:
            A, B, mask = cd_augs(batch['A'], batch['B'], batch['mask'])
            img_concat = torch.concat([A, B], dim=1)
            model_kwargs = dict(bi_img=img_concat,)
            yield mask, model_kwargs


def create_argparser():
    defaults = dict()
    defaults.update(cd_model_and_diffusion_defaults())

    defaults_ = dict(
        data_name="LEVIR",
        split='train',
        out_dir='../out_dir/zht_levir_train_V6_e60_b2',
        model_name='V6',
        lr=1e-4,
        lr_anneal_steps=600000,
        batch_size=2,
        log_interval=100,
        save_interval=100000,
        resume_checkpoint="",
        img_size=256,
        num_workers=2,
        learn_sigma=False,
        schedule_sampler='uniform',
    )
    defaults.update(defaults_)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
