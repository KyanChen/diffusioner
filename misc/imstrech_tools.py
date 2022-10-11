from utils import im2arr, save_image
import os
import numpy as np


def cStresh(img, min_=0, max_=255, mask=None, type='uint8'):
    """
    针对图像的一个通道进行预处理操作。只对mask所指定的区域拉伸操作.
    Args:
      img: 原图像的一个通道
      mask: 一个拼接区域的mask
    Returns:
      预处理后的该通道的图像
    """
    assert img.ndim == 2
    # assert img.dtype == np.uint16
    if mask is None:
        mask = np.ones([img.shape[0], img.shape[1]], dtype=np.bool)
    # print(min_, max_)
    t = img.astype(np.float32)
    maxv = max_
    minv = min_

    t[mask] = (t[mask] - minv) / (maxv - minv)  # 拉伸
    # meanv = t[mask].mean()
    # t[t < 0] = 0
    # t[t > 1] = 1
    ret = None
    if type == 'uint8':
        t *= 255
        t[t < 0] = 0
        t[t >= 255] = 255
        ret = t.astype(np.uint8)
    elif type == 'uint16':
        t *= 65536
        t[t < 0] = 0
        t[t >= 65536] = 65536
        ret = t.astype(np.uint16)

    return ret


def imMultiStresh(img, min_=0, max_=255, mask=None, type='uint8'):
    """
    实现多类mask的图像拉伸功能
    """
    K = mask.max()
    ret = np.array(img, dtype=np.uint8)
    for i in range(1, K+1):
        ret[mask==i] = cStresh(img,
                               mask=(mask==i), type=type,
                               min_=min_, max_= max_
                               )[mask==i]
    return ret


def imstresh(img, mask=None, type='uint8', min_vals=[0], max_vals=[255]):
    """
    输入多通道/单通道图像，输出图像拉伸结果
    :param img: ndarray, H*W*C
    :param mask: ndarray, H*W, bool
    :param type: return mode: uint8/uint16
    """
    assert isinstance(min_vals, list)
    if mask is None:
        mask = np.ones([img.shape[0], img.shape[1]], dtype=np.bool)
    ret = img.copy()
    if type == 'uint8':
        ret = np.array(ret, dtype=np.uint8)
    elif type == 'uint16':
        ret = np.array(ret, dtype=np.uint16)
    if img.ndim == 3:
        C = img.shape[2]
        if len(min_vals) == 1:
            min_vals = [min_vals[0], ] * C
        if len(max_vals) == 1:
            max_vals = [max_vals[0], ] * C
        for i in range(C):
            ret[:, :, i] = imMultiStresh(
                img[:, :, i], mask=mask, type=type,
                min_=min_vals[i], max_=max_vals[i]
            )
    elif img.ndim == 2:
        ret = imMultiStresh(img, mask=mask, type=type, min_=min_vals[0], max_=max_vals[0])
    return ret


