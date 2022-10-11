import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, UNetCDModel, UNetCDModel2, UNetCDModel4, \
    UNetCDModel5, UNetCDModel6



def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def cdmodel_and_diffusion_defaults():
    """
    Defaults for cd training.
    """
    res = dict(
        image_size=256,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        model_name='V1',
    )
    res.update(diffusion_defaults())
    return res


def cd_create_model_and_diffusion(
    model_name='V1',
    image_size=256,
    learn_sigma=False,
    num_channels=64,
    num_res_blocks=2,
    channel_mult="",
    num_heads=4,
    num_head_channels=-1,
    num_heads_upsample=-1,
    attention_resolutions="16",
    dropout=0,
    diffusion_steps=1000,
    noise_schedule='linear',
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    model = create_cd_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        model_name=model_name,
    )
    diffusion = create_gaussian_diffusion(
        diffusion_steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion



def create_cd_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    model_name='V1',
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    kwargs = dict(image_size=image_size,
            in_channels=7,
            model_channels=num_channels,
            out_channels=(1 if not learn_sigma else 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,)
    if model_name == 'V1':
        kwargs['in_channels'] = 7
        return UNetCDModel(
            **kwargs
            )
    elif model_name == 'V2':
        print("model v2")
        kwargs['in_channels'] = 1
        return UNetCDModel2(
            **kwargs
            )
    elif model_name == 'V2_d32':
        print("model V2_d32")
        kwargs['in_channels'] = 1
        kwargs['model_channels'] = 32

        return UNetCDModel2(
            **kwargs
            )
    elif model_name == 'V4':
        print("model v4")
        kwargs['in_channels'] = 1
        return UNetCDModel4(
            **kwargs
            )
    elif model_name == 'V5':
        print("model v5")
        kwargs['in_channels'] = 1
        return UNetCDModel5(
            **kwargs
            )
    elif model_name == 'V6':
        print("model v6")
        kwargs['in_channels'] = 1
        return UNetCDModel6(
            **kwargs
            )
    else:
        raise NotImplementedError


def create_basic_cd_model(image_size=256,
                            attention_resolutions='16'
                          ):
    from guided_diffusion.unet_basic import UNetModel as BasicUNetModel
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    return BasicUNetModel(
        image_size=image_size,
        in_channels=6,
        model_channels=64,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=attention_ds,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_heads=4,)


def cd_model_and_diffusion_defaults():
    res = cdmodel_and_diffusion_defaults()
    arg_names = inspect.getfullargspec(cd_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps) #设置扩散过程中给定的方差
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
