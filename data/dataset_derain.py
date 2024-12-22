import random
import os
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class DatasetDERAIN(data.Dataset):
    def __init__(self, opt):
        super(DatasetDERAIN, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        self.files = {}

        # ------------------------------------
        # get paths of gt/rain
        # ------------------------------------

        # RF
        self.paths_H = os.path.join(opt['dataroot_H'], opt['phase'])
        self.paths_rain = os.path.join(opt['dataroot_RAIN'], opt['phase'])

        self.files[opt['phase']] = self.recursive_glob(rootdir=self.paths_rain, suffix='.png')

        self.pairs = self._create_pairs()

        # RFS/RFD
        # self.paths_H = opt['dataroot_H']
        # self.paths_rain = opt['dataroot_RAIN']
        #
        # self.files[opt['phase']] = self.recursive_glob(rootdir=self.paths_rain, suffix='.png')
        #
        #
        # self.pairs = self._create_pairs()

        assert self.paths_H, 'Error: H path is empty.'

    # RF
    def _create_pairs(self):
        pairs = []
        for city_folder in os.listdir(self.paths_rain):
            city_rain_path = os.path.join(self.paths_rain, city_folder)
            city_clear_path = os.path.join(self.paths_H, city_folder)
            for rain_image in os.listdir(city_rain_path):
                base_name = '_'.join(rain_image.split('_')[:-9])
                clean_img_name = base_name + '.png'
                pairs.append((os.path.join(city_rain_path, rain_image), os.path.join(city_clear_path, clean_img_name)))
        return pairs

    # RFS/RFD
    # def _create_pairs(self):
    #     pairs = []
    #     for rain_image in os.listdir(self.paths_rain):
    #         clean_img_name = rain_image
    #         pairs.append((os.path.join(self.paths_rain, rain_image), os.path.join(self.paths_H, clean_img_name)))
    #     return pairs

    def __getitem__(self, index):

        L_path = None
        rain_path, H_path = self.pairs[index]


        # ------------------------------------
        # get rain image
        # ------------------------------------
        # rain_path = self.files[self.opt['phase']][index].rstrip()
        img_rain = util.imread_uint(rain_path, self.n_channels)
        img_rain = util.uint2single(img_rain)

        # ------------------------------------
        # get H image
        # ------------------------------------
        # H_path = os.path.join(self.paths_H,
        #                        rain_path.split(os.sep)[-2],
        #                        rain_path.split(os.sep)[-2] + '_' +rain_path.split('_')[1] + '_' + rain_path.split('_')[2] + '_' + rain_path.split('_')[3] + '.png')
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # --------------------------------
        # get L image & img_H_feature
        # sythesize L image via matlab's bicubic
        # --------------------------------
        H, W = img_rain.shape[:2]
        img_L = util.imresize_np(img_rain, 1 / self.sf, True)
        img_H_feature = util.imresize_np(img_H, 1 / self.sf, True)


        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch & img_H_feature patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            img_H_feature = img_H_feature[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_rain = img_rain[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 3)
            # img = img_H
            img_L, img_H, img_H_feature, img_rain = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode), util.augment_img(img_H_feature, mode=mode), util.augment_img(img_rain, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L, img_H_feature, img_rain = util.single2tensor3(img_H), util.single2tensor3(img_L), util.single2tensor3(img_H_feature), util.single2tensor3(img_rain)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'rain': img_rain, 'H_feature': img_H_feature, 'L_path': L_path, 'H_path': H_path, 'rain_path': rain_path}

    def __len__(self):
        return len(self.files[self.opt['phase']])

    def recursive_glob(self, rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]
