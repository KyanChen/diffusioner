
from guided_diffusion.unet import UNetCDModel2

if __name__ == '__main__':
    import torch

    device = torch.device('cuda:0')
    model = UNetCDModel2(image_size=256,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[16, ],
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),).to(device)

    bi_img = torch.randn([2, 6, 256, 256], dtype=torch.float32).to(device)
    data_in = torch.randn([2, 1, 256, 256], dtype=torch.float32).to(device)
    timesteps = torch.FloatTensor([1]).to(device)
    data_out = model(data_in, timesteps, bi_img)

    print(type(data_out))
    if isinstance(data_out, tuple):
        print('shape of the output: ', data_out[0].shape)
    else:
        print('shape of the output: ', data_out.shape)
