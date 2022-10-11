#  _*_ coding: utf-8 _*_

"""
ref: https://blog.csdn.net/XBR_2014/article/details/85141830
该工具用于提取遥感数据的地理信息，获取经纬度信息，以及重投影操作，转换坐标系操作；
"""

import numpy as np
from osgeo import gdal, osr
from utils import fresize,im2arr,save_image
import misc.gdal_utils as gdal_utils


def get_geo_epsg(image_path):
    """获取tif图像的ESPG代号"""
    proj = gdal_utils.get_projection(image_path)
    index = proj.rfind('"EPSG",')
    epsg_str = 'EPSG:' + proj[index+7:].split('"')[1]
    print(epsg_str)
    return epsg_str

def getBasicInfo(dataset):
    print('数据投影：')
    print(dataset.GetProjection())
    print('数据的大小（行，列）：')
    print('(%s %s)' % (dataset.RasterYSize, dataset.RasterXSize))


def getSRSPair(dataset):
    """
    ref ：https://blog.csdn.net/theonegis/article/details/54427906
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    """
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    # print(prosrs, geosrs)
    return prosrs, geosrs

def geo2lonlat(dataset, x, y):
    """
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)

    """
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    ## case1:
    # coords = ct.TransformPoint(x, y)
    # lon, lat = coords[:2]
    # WARNNING! TODO: 这里有个bug，有些位置输入(x,y)时计算正确，有些位置(y,x)
    # coords 是一个Tuple类型的变量包含3个元素，coords [0]为纬度，coords [1]为经度，coords [2]为高度
    # 这里比较奇怪的一点是，返回的坐标，是先纬度后经度,    #  也有可能，返回的坐标，是先经度，后纬度
    # case2
    coords = ct.TransformPoint(y, x)
    lat, lon = coords[:2]


    return (lon, lat)


def lonlat2geo(dataset, lon, lat):
    """
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(geo_x, geo_y)对应的投影坐标
    """
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lat, lon)
    geo_y, geo_x = coords[:2]
    return (geo_x, geo_y)

def imagexy2geo(dataset, x, y):
    """
     根据GDAL的六参数模型将影像图上坐标（列行号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param y: 像素的行号
    :param x: 像素的列号
    :return: 列行号(x, y)对应的投影坐标或地理坐标(geo_x, geo_y)
    """
    trans = dataset.GetGeoTransform()

    geo_x = trans[0] + x * trans[1] + y * trans[2]
    geo_y = trans[3] + x * trans[4] + y * trans[5]
    # print(geo_x, geo_y)
    return geo_x, geo_y


def geo2imagexy(dataset, x, y):
    """
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上列行号(col, row)
    """
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def imagexy2lonlat(dataset,col, row):
    geo_x, geo_y = imagexy2geo(dataset, col, row)
    lon, lat = geo2lonlat(dataset, geo_x, geo_y)
    return lon, lat


def lonlat2imagexy(dataset, lon, lat):
    geo_x, geo_y = lonlat2geo(dataset, lon, lat)
    x, y = geo2imagexy(dataset, geo_x, geo_y)
    return x, y


def getGEOBounds(dataset):
    """获取数据的地理坐标的边界值；
    :param dataset: gdal_dataset
    :return: [左，右，上，下]
    """
    trans = dataset.GetGeoTransform()
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    pxs = []
    pys = []
    for col in [0, width]:
        for row in [0, height]:
            px = trans[0] + col * trans[1] + row * trans[2]
            py = trans[3] + col * trans[4] + row * trans[5]
            pxs.append(px)
            pys.append(py)
    print(min(pxs), max(pxs), min(pys), max(pys))
    return [min(pxs), max(pxs), min(pys), max(pys)]


def getLonLatBounds(dataset):
    """获取数据的经纬度坐标的边界值；
    :param dataset: gdal_dataset
    :return: [左，右，下,上]
    """
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    pxs = []
    pys = []
    for x in [0, width]:
        for y in [0, height]:
            geo_x, geo_y = imagexy2geo(dataset, x, y)
            lon, lat = geo2lonlat(dataset, geo_x, geo_y)
            # print(lon, lat)
            pxs.append(lon)
            pys.append(lat)
    return [min(pxs), max(pxs), min(pys), max(pys)]


def cal_overlap_xy(ds, ref_ds):
    """认为ref_ds的地理范围小于ds，从需要从ds中截取一部分,返回截取的图像像素位置
    :param ds: gdal dataset
    :param ref_ds: gdal dataset
    :return: ndaray, dtype=int, [x1, y1, x2, y2]，
    注意：x方向为行方向，y方向为列方向；与GDAl的默认方向一致；
    """
    trans = ds.GetGeoTransform()
    print('trans:', trans)
    trans_ref = ref_ds.GetGeoTransform()
    print('trans_ref:', trans_ref)

    height = ref_ds.RasterYSize
    width = ref_ds.RasterXSize
    # 获取左上角的相对位置
    px, py = imagexy2geo(ref_ds, 0, 0)
    print('ref,(0,0)->(px,py): ', px, py)
    x1, y1 = geo2imagexy(ds, px, py)
    print('ds,(0,0)->(x,y): ', x1, y1)
    # 获取右下角的相对位置
    px, py = imagexy2geo(ref_ds, height, width)
    print('ref,(', height,',',width,')->(px,py): ', px, py)
    x2, y2 = geo2imagexy(ds, px, py)
    print('ds,(height,width)->(x,y): ', x2, y2)
    # box = np.array([x1, y1, x2, y2], dtype=np.int)
    return [int(round(x1)), int(round(y1)),
            int(round(x2)), int(round(y2))]

def cutResizeROI(im_path, ref_path, save_path):
    """
    根据参考图像的地理范围，对im做剪裁，并拉伸到相同分辨率；
    注意：受剪裁图像的地理范围需要大于参考图像；
    :param im_path: str,被剪裁图像路径
    :param ref_path: str, 参考图像路径
    :param save_path: str, 剪裁图像保存路径
    """
    in_ds = gdal.Open(im_path)
    ref_ds = gdal.Open(ref_path)
    height = ref_ds.RasterYSize
    width = ref_ds.RasterXSize
    box = cal_overlap_xy(ds=in_ds, ref_ds=ref_ds)
    print(box)
    im = im2arr(im_path)
    # out = im[box[1]:box[1]+height, box[0]:box[0]+width, :]
    out = im[box[1]:box[3], box[0]:box[2], :]
    out = fresize(out, [height, width])
    save_image(out, save_path)


def crop_roi(image_path, out_path, x, y, crop_x_size, crop_y_size, scale_factor=1):
    im_height, im_width, _ = gdal_utils.get_image_shape(image_path)
    geo_transforms = gdal_utils.get_geoTransform(image_path)
    proj = gdal_utils.get_projection(image_path)
    roi_image = gdal_utils.read_image(image_path, x, y, crop_x_size, crop_y_size, scale_factor=scale_factor, as_rgb=False,
               data_format='GDAL_FORMAT')

    crop_geotransforms = gdal_utils.update_transform(geo_transforms, x, y)

    gdal_utils.save_full_image(out_path, roi_image, geoTranfsorm=crop_geotransforms,
                               proj=proj)

def crop_roi_and_resize(image_path, out_path, x, y, crop_x_size, crop_y_size,
                        target_w, target_h):
    from misc.imutils import pil_resize
    im_height, im_width, _ = gdal_utils.get_image_shape(image_path)
    geo_transforms = gdal_utils.get_geoTransform(image_path)
    proj = gdal_utils.get_projection(image_path)
    roi_image = gdal_utils.read_image(image_path, x, y, crop_x_size, crop_y_size,
                                      scale_factor=1,
                                      as_rgb=False,
                                      data_format='NUMPY_FORMAT')
    if crop_x_size != target_w or crop_y_size != target_h:
        roi_image = pil_resize(roi_image, size=(target_h, target_w), order=3)

    crop_geotransforms = gdal_utils.update_transform(geo_transforms, x, y)

    gdal_utils.save_full_image(out_path, roi_image, geoTranfsorm=crop_geotransforms,
                               proj=proj, data_format="NUMPY_FORMAT")

def cut_roi_by_ref(im_path, ref_path, save_path):
    """
    根据参考图像的地理范围，对im做剪裁，到相同分辨率；
    注意：受剪裁图像的地理范围需要大于参考图像；
    :param im_path: str,被剪裁图像路径
    :param ref_path: str, 参考图像路径
    :param save_path: str, 剪裁图像保存路径
    """
    in_ds = gdal.Open(im_path)
    ref_ds = gdal.Open(ref_path)
    im_height, im_width, im_bands = gdal_utils.get_image_shape(ref_path)
    x1, y1, x2, y2 = cal_overlap_xy(ds=in_ds, ref_ds=ref_ds)
    # assert x2-x1 == im_width
    # assert y2-y1 == im_height
    # crop_roi(im_path, out_path=save_path, x=x1, y=y1,
    #          crop_x_size=x2-x1, crop_y_size=y2-y1)
    crop_roi_and_resize(im_path, out_path=save_path, x=x1, y=y1,
                        crop_x_size=x2-x1, crop_y_size=y2-y1,
                        target_h=im_height, target_w=im_width)


def ds2WGS84ds(ds):
    """
    把任意gdal dateset转化为WGS84下的dataset
    :param ds: gdal dataset
    :return: vrt_ds: gdal dataset
    """
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    vrt_ds = gdal.AutoCreateWarpedVRT(ds, None, srs.ExportToWkt(), gdal.GRA_Bilinear)
    return vrt_ds


def resample(im_path,out_path,srs='EPSG:26914',xRes=0.3,yRes=0.3):
    """遥感数据的重投影，指定目标投影坐标系，
    将原始数据（代地理坐标系）投影到该系；
    注意：源dataset与目标dataset之间，坐标单位要一致，比如都是m或者都是经纬度，否则报错。
    另外：小用法，如果想要知道目标的EPSG代号，可以在qgis软件中打开，查看坐标参考系，然后百度一下对应的代号
    利用gdalwarp对map做重投影
        世界坐标系统，查询网站
        ref:  http://epsg.io/
        常用对应关系，（坐标名，球体/大地水准面）--EPSG代号
        NAD83 / UTM zone 14N: EPSG:26914
        WGS84: EPSG:4326
        WGS 84 / UTM zone 49N - EPSG:32649
        EPSG:3857 (Pseudo-Mercator)(Web 墨卡托投影)
        EPSG:2154 RGF93 / Lambert-93 - France
        2193
    :param im_path: str
    :param out_path: str
    :param srs: EPSG代号
    :param xRes: 重采样的x方向分辨率 (与6元素中尺度保持相同的单位)
    :param yRes: 重采样的x方向分辨率
    """
    in_ds = gdal.Open(im_path)
    gdal.Warp(out_path, in_ds, xRes=xRes, yRes=yRes, dstSRS=srs)



def resample_v2(im_path, save_path):
    # 利用VRT方式做重投影，重投影到经纬度坐标系下
    #  输出投影方式
    # 可以适用于源dataset与目标dataset中，坐标单位可以不一样。

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    old_ds = gdal.Open(im_path)
    vrt_ds = gdal.AutoCreateWarpedVRT(old_ds, None, srs.ExportToWkt(), gdal.GRA_Bilinear)
    gdal.GetDriverByName('gtiff').CreateCopy(save_path, vrt_ds)


# 利用VRT方式做重投影
#  输出投影方式
# srs = osr.SpatialReference()
# srs.SetWellKnownGeogCS('WGS84')
# old_ds = gdal.Open(im_path)
# vrt_ds = gdal.AutoCreateWarpedVRT(old_ds, None, srs.ExportToWkt(), gdal.GRA_Bilinear)
# gdal.GetDriverByName('gtiff').CreateCopy(save_path, vrt_ds)


