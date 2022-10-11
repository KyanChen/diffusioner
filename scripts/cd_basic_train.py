"""
Train a change detection model.
"""
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import argparse

import torch.nn.functional as F
import torch
from guided_diffusion import logger
from datasets import get_loader
from guided_diffusion.script_util import (
    create_basic_cd_model,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import BasicCDTrainLoop


def main():
    args = create_argparser().parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    out_dir = args.out_dir
    logger.configure(dir=out_dir)

    logger.log("creating model...")

    model = create_basic_cd_model()
    model.to(device)

    logger.log("creating data loader...")
    data = load_cd_data(
        args.data_name,
        args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        split=args.split,
    )

    logger.log("training...")
    BasicCDTrainLoop(
        model=model,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
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
            # model_kwargs = dict(bi_img=img_concat,)
            yield img_concat, mask


def create_argparser():
    defaults = dict(
        data_name="LEVIR",
        split='trainval1',
        out_dir='../out_dir/levir_V1_e20',
        lr=1e-4,
        lr_anneal_steps=200000,
        batch_size=4,
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        img_size=256,
        num_workers=2,
        use_fp16=False,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
