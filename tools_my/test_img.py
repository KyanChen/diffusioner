import torch
import math
import torch as th
import torch.nn as nn
# img = torch.randn([2, 1, 256, 256])
# print(img)
# img = torch.sigmoid(img)
# img[img>=0.5] = 1
# img[img<0.5] = 0
# # img = torch.sigmoid(img)
# print(img >= 0.5)
# print(img)
#
# img1 = torch.randn([2, 1, 256, 256]) #prelable
# img2 = 1 - img1
# img = torch.cat([img1, img2], dim=1)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

if __name__ == "__main__":
    b, c, *_spatial = x.shape
    x = x.reshape(b, c, -1)  # NC(HW)
    x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
    x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
    x = self.qkv_proj(x)

    positional_embedding = nn.Parameter(
        th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
    )
    qkv = QKVAttention(8)
    img = torch.randn([2, 256, 16, 16])
    img = img.reshape(b, c, -1)
    img = th.cat([img.mean(dim=-1, keepdim=True), img], dim=-1)
    img = img + positional_embedding[None, :, :].to(x.dtype)