import os


class DataConfig(dict):
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    label_folder_name = 'label'
    img_folder_name = ['A']
    img_folder_names = ['A', 'B']
    n_class = 2
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = r'E:\cddataset\LEVIR-CD256'
            # self.root_dir = r'R:\levir256'
        elif data_name == 'LEVIRORI':
            self.root_dir = r'D:\dataset\CD\LEVIR-CD\ORI_LARGE\cut\cut'
        elif data_name == 'LEVIRORIMAP':
            self.root_dir = r'D:\dataset\CD\LEVIR-CD\ORI_LARGE\cut\cut'
            self.label_folder_name = 'map_building_2111'
        elif data_name == 'LEVIRMAP':
            self.root_dir = 'D:/dataset/CD/LEVIR-CD/cut/'
            self.n_class = 6
            self.label_folder_name = 'label_map'
        elif data_name == 'WHU':
            self.root_dir = r'E:\cddataset\WHU-CD-256'
        elif data_name == 'google':
            self.root_dir = r'D:\dataset\CD\SemiCNet\CD_Data_GZ\cut'
        elif data_name == 'loveda':
            self.root_dir = r'D:\dataset\segdata\2021LoveDA\Train\cut_256'
        elif data_name == 'xbd':
            self.root_dir = r'D:\dataset\CD\xBD\cut_256_pos'
        elif data_name == 'gan':
            self.root_dir = r'D:\dataset\CD\gan-paper'
        elif data_name == 'sysu':
            self.root_dir = r'D:\dataset\CD\SYSU-CD'
        elif data_name == 'DSIFN':
            self.root_dir = r'D:\dataset\CD\DSIFN-Dataset\cut256'
        elif data_name == 'DSIFN_512':
            self.root_dir = r'D:\dataset\CD\DSIFN-Dataset\origin'
        elif data_name == 'Seg_LEVIRMAP':
            self.root_dir = r'D:\dataset\CD\RS_Mapping\cut256'
            self.n_class = 5
            self.label_transform = 'ignore0_sub1'
        elif data_name == 'inria':
            self.root_dir = r'I:\data\segmentation\Inria\cut'
        elif data_name == 'inria256':
            self.root_dir = r'G:\tmp_data\inria_cut256'
            # self.root_dir = r'R:\inria256'
        elif data_name == 'airs':
            self.root_dir = r'I:\data\segmentation\AIRS\trainval\train\cut'
            self.n_class = 2
        elif data_name == 'airs256':
            self.root_dir = r'G:\tmp_data\airs256'
            # self.root_dir = r'R:\airs256'
            self.n_class = 2
        elif data_name == 'LEVIR+NC':
            self.root_dir = r'D:\dataset\CD\LEVIR-CD+\other_collect\cut'
        elif data_name == 'LEVIR+ALL':
            self.root_dir = r'G:\tmp_data\levir+all_cut'
        elif data_name == 'minifranceMAP':
            self.root_dir = r'D:\dataset\segdata\MiniFrance-suite\cut_256'
            self.label_folder_name = 'map_building_21'
            self.n_class = 2
        elif data_name == 'spacenet2':
            self.root_dir = r'G:\tmp_data\spacenet2'
            self.n_class = 2
        elif data_name == 'spacenet3':
            self.root_dir = r'G:\tmp_data\spacenet3d'
            self.n_class = 2
        elif data_name == 'spacenet6':
            self.root_dir = r'G:\tmp_data\spacenet6'
            self.n_class = 2
        elif data_name == 'SLADCD256':
            self.root_dir = 'D:\dataset\CD\SLADCD\cut'
        elif data_name == 'SLADCD512':
            self.root_dir = r'G:\tmp_data\SLADCD\cut'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


def get_pretrained_path(pretrained):
    out = None
    if pretrained is not None:
        if os.path.isfile(pretrained):
            out = pretrained
        elif pretrained == 'imagenet':
            out = pretrained
        elif pretrained == 'intern_res18':
            out = r'G:\program\CD\CD4_2\pretrain\INTERN-R18-e6b9ff7cb.pth'
        elif pretrained == 'intern_res50':
            out = r'G:\program\CD\CD4_2\pretrain\INTERN-R50-7e13cd38c.pth'
        elif pretrained == 'detcon_100':
            out = r'G:\program\CD\CD4_2\pretrain\detcon_100.pth'

        elif pretrained == 'geokr_res50':
            out = r'G:\program\CD\CD4\pretrain\GeoKR\resnet50_.pth'
        elif pretrained == 'geokr2_res50':
            out = r'G:\program\CD\CD4\pretrain\GeoKR\resnet50_2.pth'
        elif pretrained == 'landmap_res18':
            out = r'G:\program\CD\CD4\pretrain\LandMap\resnet18_2.pth'
        elif pretrained == 'landmap_res50':
            out = r'G:\program\CD\CD4\pretrain\LandMap\resnet50_.pth'
        elif pretrained == 'swav_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\swav.pth'
        elif pretrained == 'deepclusterv2_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\deepcluster-v2.pth'
        elif pretrained == 'mocov2_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\moco_v2.pth'
        elif pretrained == 'pclv2_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\pcl-v2.pth'
        elif pretrained == 'infomin_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\infomin.pth'
        elif pretrained == 'simclrv2_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\simclr-v2.pth'
        elif pretrained == 'densecl_res50':
            out = r'G:\program\CD\CD4\pretrain\self-sup\densecl_r50_imagenet_200ep.pth'
        elif pretrained == 'seco_resnet18_100k':
            out = r'G:\program\CD\CD4_1\pretrain\seco_resnet18_100k.pth'
        elif pretrained == 'seco_resnet18_1m':
            out = r'G:\program\CD\CD4_1\pretrain\seco_resnet18_1m.pth'
        elif pretrained == 'seco_resnet18':
            out = r'G:\program\CD\CD4_1\pretrain\seco_resnet18.pth'
        elif pretrained == 'inria_seg':
            out = r'G:\program\CD\CD4_1\pretrain\inria_seg.pth'
        elif pretrained == 'airs_seg':
            out = r'G:\program\CD\CD4_1\pretrain\airs_seg.pth'

        elif pretrained == 'None' or pretrained == 'none':
            out = None
        else:
            raise NotImplementedError(pretrained)
    else:
        out = None
    return out


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='Seg_LEVIRMAP')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)
    print(data.n_class)

