import os
from misc.pyutils import mkdir, get_paths
from misc.imutils import save_image, im2arr
import misc.gdal_utils as gdal_utils
import math
import numpy as np



########################################basic funcs#############################################################

def checkBoundry(x, cropsize, right):
    x_left = x
    x_right = x + cropsize
    if x + cropsize > right:
        x_right = right
        x_left = right - cropsize
    #TODO: if x_left <0:
    return x_left, x_right


def crop_img_with_geo(img_paths, dest_folder, sampleSize=256, stepsize=256, drop=False, compress=None):
    for img_path in img_paths:

        basename = os.path.basename(img_path).split('.')[0]
        print('process: ', basename)
        A_path = img_path

        im_height, im_width, _ = gdal_utils.get_image_shape(A_path)
        geo_transforms = gdal_utils.get_geoTransform(A_path)
        projection = gdal_utils.get_projection(A_path)

        h, w = im_height, im_width
        # 如果舍弃最后一个不完整切片
        if drop:
            h_ = sampleSize + (math.floor((h - sampleSize) / stepsize)) * stepsize
            w_ = sampleSize + (math.floor((w - sampleSize) / stepsize)) * stepsize
        else:
            h_ = sampleSize + (math.ceil((h - sampleSize) / stepsize)) * stepsize
            w_ = sampleSize + (math.ceil((w - sampleSize) / stepsize)) * stepsize

        for i in range(0,h_,stepsize):
            for j in range(0,w_,stepsize):
                y1, y2= checkBoundry(i,sampleSize,h)
                x1, x2= checkBoundry(j,sampleSize,w)

                patch_A = gdal_utils.read_image(A_path, x1, y1, sampleSize, sampleSize, scale_factor=1, as_rgb=False,
                                            data_format='NUMPY_FORMAT')

                name = "%s_%04d_%04d.tif" % (basename, i, j)

                # n_A = (patch_A == 0).sum()/(patch_A.shape[0]*patch_A.shape[1]*patch_A.shape[2])
                # if n_A >= 0.1:  # 跳过黑边(黑边占比大于threshold时，不考虑该切片)
                #     continue
                # n_A = (patch_A == 255).sum()/(patch_A.shape[0]*patch_A.shape[1]*patch_A.shape[2])
                # if n_A >= 0.1:  # 跳过白边
                #     continue

                save_path = os.path.join(dest_folder, name)
                crop_geotransforms = gdal_utils.update_transform(geo_transforms, x1, y1)
                gdal_utils.save_full_image(save_path, patch_A, geoTranfsorm=crop_geotransforms,
                                           proj=projection, data_format='NUMPY_FORMAT',
                                           compress=compress)


def crop_img(img_paths, dest_folder, sampleSize=256, stepsize=256, drop=False, out_suffix='.png',
             crop_name_mode='h_w', out_size=None):
    if out_size is None:
        out_size = sampleSize
    for img_path in img_paths:

        basename = os.path.basename(img_path).split('.')[0]
        print('process: ', basename)
        A_path = img_path

        im_height, im_width, _ = gdal_utils.get_image_shape(A_path)

        h, w = im_height, im_width
        # 如果舍弃最后一个不完整切片
        if drop:
            h_ = sampleSize + (math.floor((h - sampleSize) / stepsize)) * stepsize
            w_ = sampleSize + (math.floor((w - sampleSize) / stepsize)) * stepsize
        else:
            h_ = sampleSize + (math.ceil((h - sampleSize) / stepsize)) * stepsize
            w_ = sampleSize + (math.ceil((w - sampleSize) / stepsize)) * stepsize

        for i in range(0,h_,stepsize):
            for j in range(0,w_,stepsize):
                y1, y2= checkBoundry(i,sampleSize,h)
                x1, x2= checkBoundry(j,sampleSize,w)

                patch_A = gdal_utils.read_image(A_path, x1, y1, sampleSize, sampleSize, scale_factor=1, as_rgb=False,
                                            data_format='NUMPY_FORMAT')
                if patch_A.ndim == 3:
                    if patch_A.shape[-1] == 1:
                        # 适配label
                        patch_A = patch_A[:,:,0]

                resample = 3
                if patch_A.ndim==2 and patch_A.max()==1:
                    patch_A *= 255
                    resample = 0
                if out_size != sampleSize:
                    from misc.imutils import pil_resize
                    patch_A = pil_resize(patch_A, size=(out_size, out_size), order=resample)
                if crop_name_mode == 'h_w':
                    name = "%s_%04d_%04d" % (basename, i, j) + out_suffix
                elif crop_name_mode == 'w_h':
                    name = "%s_%04d_%04d" % (basename, j, i) + out_suffix
                else:
                    raise NotImplementedError(crop_name_mode)
                # n_A = (patch_A == 0).sum()/(patch_A.shape[0]*patch_A.shape[1]*patch_A.shape[2])
                # if n_A >= 0.1:  # 跳过黑边(黑边占比大于threshold时，不考虑该切片)
                #     continue
                # n_A = (patch_A == 255).sum()/(patch_A.shape[0]*patch_A.shape[1]*patch_A.shape[2])
                # if n_A >= 0.1:  # 跳过白边
                #     continue

                save_path = os.path.join(dest_folder, name)
                save_image(patch_A, save_path)


def crop_single_patch(img_path, dest_folder, x1, y1, sampleSize=256, out_suffix='.png',
             crop_name_mode='h_w'):

    basename = os.path.basename(img_path).split('.')[0]
    # print('process: ', basename)
    A_path = img_path

    im_height, im_width, _ = gdal_utils.get_image_shape(A_path)

    h, w = im_height, im_width
    x1 = int(x1)
    y1 = int(y1)
    y1, y2 = checkBoundry(y1, sampleSize, h)
    x1, x2 = checkBoundry(x1, sampleSize, w)

    patch_A = gdal_utils.read_image(A_path, x1, y1, sampleSize, sampleSize, scale_factor=1, as_rgb=False,
                                data_format='NUMPY_FORMAT')
    if patch_A.ndim == 3:
        if patch_A.shape[-1] == 1:
            # 适配label
            patch_A = patch_A[:,:,0]
    if crop_name_mode == 'h_w':
        name = "%s_%04d_%04d" % (basename, y1, x1) + out_suffix
    elif crop_name_mode == 'w_h':
        name = "%s_%04d_%04d" % (basename, x1, y1) + out_suffix
    else:
        raise NotImplementedError(crop_name_mode)

    save_path = os.path.join(dest_folder, name)
    save_image(patch_A, save_path)
######################################## do funcs #############################################################


def cut_mode():
    # 定义CD数据原始待切割数据位置

    root = r'F:\data\google\Collect_data-2019-06'

    dest_folder = os.path.join(root, 'map_with_geo')

    mkdir(dest_folder)
    # 定义切割参数
    sampleSize = 1024
    stepsize = sampleSize
    img_paths = get_paths(root,'*/*/*_crop.tif')

    print(img_paths)
    crop_img_with_geo(img_paths, dest_folder=dest_folder, sampleSize=sampleSize, stepsize=stepsize,drop=True)


if __name__ == '__main__':
    cut_mode()