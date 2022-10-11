# channel_mult=(1, 2, 4, 8)
# print(type(channel_mult))
# for level, mult in enumerate(channel_mult):
#     print("level is",level)
#     print("mult is",mult)
import torch
import torch.nn as nn

# attention_resolutions=[16, ]
# bs = 1
# if bs in attention_resolutions:
#     print("zht")

from guided_diffusion.resnet import *

class ResNet(nn.Module):
    def __init__(self, resnet_stage=4, backbone='resnet_50', is_upsample_2x=True):
        super(ResNet, self).__init__()

        self.is_upsample_2x = is_upsample_2x
        expand=1

        if backbone == 'resnet_18':
            self.resnet = resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])

        elif backbone == 'resnet_34':
            self.resnet = resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])

        elif backbone == 'resnet_50':
            self.resnet = resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4

        else:
            raise NotImplementedError


        self.resnet_stage = resnet_stage

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear')
        self.avgpooling = nn.AdaptiveAvgPool2d((8, 8))

        if self.resnet_stage == 5:
            layers = 512 * expand
        elif self.resnet_stage == 4:
            layers = 256 * expand
        elif self.resnet_stage == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError

        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_2 = self.resnet.maxpool(x) # 1/4 in=3, out=64

        x_4 = self.resnet.layer1(x_2) # 1/4 in=64, out=64

        x_8 = self.resnet.layer2(x_4) # 1/8 in=64, out=128

        if self.resnet_stage > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8 in=128, out=256
            x_8 = self.avgpooling(x_8)

        if self.resnet_stage == 5:
            x_8 = self.resnet.layer4(x_8) # 1/8 in=256, out=512

        elif self.resnet_stage > 5:
            raise NotImplementedError
        x = x_8

        return x


if __name__ == "__main__":

    resnet = ResNet(backbone="resnet_18").to("cuda:0")
    img = torch.randn([2, 3, 256, 256]).to("cuda:0")

    res = resnet(img)

    print("res shape is", res.shape)