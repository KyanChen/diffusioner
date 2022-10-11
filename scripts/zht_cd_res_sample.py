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
#seed 3407 3408
th.manual_seed(3408)            # 为CPU设置随机种子
th.cuda.manual_seed(3408)      # 为当前GPU设置随机种子
th.cuda.manual_seed_all(3408)  # 为所有GPU设置随机种子

from guided_diffusion import logger
from guided_diffusion.script_util import (
    cd_model_and_diffusion_defaults,
    cd_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from misc.metric_tool import ConfuseMatrixMeter
from collections import OrderedDict
from datasets import get_loader
import logging
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

    metric = ConfuseMatrixMeter(n_class=2)
    log_dict = OrderedDict()
    # logger_z = logging.getLogger('base')
    # logger_test = logging.getLogger('test')

    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        mask, model_kwargs, name = next(data)
        print("mask shale is",mask.shape)
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

        print("sample shape ", sample.shape)
        ##caclute acc
        gt = mask.to(device).long()

        # pred score
        G_pred = sample.detach()
        # G_pred = torch.argmax(G_pred, dim=1)
        # print(type(G_pred[0][0]))
        # print(G_pred[0][0].type)
        # G_pred = torch.sigmoid(G_pred)
        G_pred[G_pred > 0.5] = 1
        G_pred[G_pred <= 0.5] = 0
        G_pred = G_pred.int()

        current_score = metric.update_cm(pr=G_pred.cpu().numpy(), gt=gt.detach().cpu().numpy())
        log_dict['running_acc'] = current_score.item()
        logs = log_dict
        message = '[Testing CD]. running_mf1: %.5f\n' % (logs['running_acc'])
        logger.log(message)


        out_color = False
        if not out_color:
            sample = (sample > 0.5).to(th.uint8)
        sample = sample[:, 0]
        all_images.append(sample.cpu().numpy())
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
    scores = metric.get_scores()
    epoch_acc = scores['mf1']
    log_dict['epoch_acc'] = epoch_acc.item()
    for k, v in scores.items():
        log_dict[k] = v
    logs = log_dict
    message = '[Test CD summary]: Test mF1=%.5f \n' % \
              (logs['epoch_acc'])
    for k, v in logs.items():
        message += '{:s}: {:.4e} '.format(k, v)
        message += '\n'
    logger.log(message)


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
        # print("mask shape is", mask.shape)
        # print("batch['name'] is", batch['name'])
        yield mask, model_kwargs, batch['name']


def create_argparser():
    defaults = dict()
    defaults.update(cd_model_and_diffusion_defaults())
    defaults_ = dict(
        data_name="LEVIR", #LEVIR  WHU
        model_name='V5',
        split='test',
        out_dir='../out_dir/zht_levir_train_V5_e60_b2/test_e60_step75_3',
        learn_sigma=False,
        img_size=256,
        clip_denoised=True,
        num_samples=2048, #744 2048
        batch_size=16,
        diffusion_steps=1000,
        timestep_respacing='75',  # ddim25
        use_ddim=False,
        base_samples="",
        model_path="../out_dir/zht_levir_train_V5_e60_b2/model600000.pt",
    )
    defaults.update(defaults_)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
