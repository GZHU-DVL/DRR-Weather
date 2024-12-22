import os.path
import argparse
import numpy as np
import logging
import random
import math
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, Sampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

torch.set_num_threads(2)


class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, seed=None, shuffle=True, batch_size=1, drop_last=False):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0

        self.distributed_sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            seed=seed
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.distributed_sampler.set_epoch(epoch)

    def _create_batches(self):
        indices = list(self.distributed_sampler)
        pairs = [self.dataset.pairs[i] for i in indices]

        if self.shuffle:
            random.shuffle(pairs)

        batches = []
        batch, backgrounds_seen = [], set()
        deferred_indices = []

        for pair in pairs:
            background = '_'.join(os.path.basename(pair[0]).split('_')[:3])
            if background not in backgrounds_seen:
                backgrounds_seen.add(background)
                batch.append(self.dataset.pairs.index(pair))
                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch, backgrounds_seen = [], set()
            else:
                deferred_indices.append(pair)

        if batch and not self.drop_last:
            batches.append(batch)

        while deferred_indices:
            batch, backgrounds_seen = [], set()
            temp = deferred_indices[:]
            for idx in temp:
                background = '_'.join(os.path.basename(idx[0]).split('_')[:3])
                if background not in backgrounds_seen:
                    batch.append(self.dataset.pairs.index(idx))
                    backgrounds_seen.add(background)
                    deferred_indices.remove(idx)
                if len(batch) == self.batch_size:
                    break
            if len(batch) == self.batch_size:
                batches.append(batch)

        return batches

    def __iter__(self):
        self.batches = self._create_batches()
        return iter(self.batches)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.num_replicas)


def main(json_path=''):

    # --------------------opt--------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # --------------------distributed settings--------------------

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # --------------------update opt--------------------

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    # --------------------configure logger--------------------

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # --------------------seed--------------------

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # --------------------creat dataloader--------------------

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set.files['train']) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set.files['train']), train_size))
            if opt['dist']:
                # RFS/RFD
                # train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                # train_loader = DataLoader(train_set,
                #                           batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                #                           shuffle=False,
                #                           num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                #                           drop_last=True,
                #                           pin_memory=True,
                #                           sampler=train_sampler)

                # RainCityscapes
                rain_sampler = CustomDistributedSampler(
                    train_set,
                    num_replicas=opt['world_size'],
                    rank=opt['rank'],
                    batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                    drop_last=True,
                    seed=seed
                )

                train_loader = DataLoader(train_set,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          pin_memory=True,
                                          batch_sampler=rain_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'val':
            val_set = define_Dataset(dataset_opt)
            val_loader = DataLoader(val_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)


    # --------------------initialize model--------------------

    model = define_Model(opt)
    # cnt_whole = sum([p.numel() for p in model.netG.parameters() if p.requires_grad])
    # params = clever_format([cnt_whole], "%.3f")
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # -------------------------------
    # test
    # -------------------------------

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_psnr_b = 0.0

    avg_psnr_y = 0.0
    avg_ssim_y = 0.0

    idx = 0

    for val_data in val_loader:
        idx += 1
        image_name_ext = os.path.basename(val_data['rain_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'])
        util.mkdir(img_dir)

        model.feed_data(val_data)
        model.test()

        visuals = model.current_visuals()
        H_img = util.tensor2uint(visuals['H'])
        E_derain = util.tensor2uint(visuals['E_derain'])

        # -----------------------
        # save estimated image E_derain
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
        util.imsave(E_derain, save_img_path)

        # -----------------------
        # calculate
        # -----------------------
        current_psnr = util.calculate_psnr(E_derain, H_img)
        current_ssim = util.calculate_ssim(E_derain, H_img)
        current_psnr_b = util.calculate_psnrb(E_derain, H_img)

        output_y = util.bgr2ycbcr(E_derain.astype(np.float32) / 255.) * 255.
        img_gt_y = util.bgr2ycbcr(H_img.astype(np.float32) / 255.) * 255.
        current_psnr_y = util.calculate_psnr(output_y, img_gt_y)
        current_ssim_y = util.calculate_ssim(output_y, img_gt_y)

        logger.info(
            '{:->4d}--> {:>10s} | PSNR : {:.4f} , SSIM : {:.4f} , PSNR_b : {:.4f} , PSNR_y : {:.4f} , SSIM_y : {:.4f}'.format(
                idx, image_name_ext, current_psnr, current_ssim, current_psnr_b, current_psnr_y,
                current_ssim_y))

        # logger.info('{:->4d}--> {:>10s}'.format(idx, image_name_ext))

        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_psnr_b += current_psnr_b

        avg_psnr_y += current_psnr_y
        avg_ssim_y += current_ssim_y

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_psnr_b = avg_psnr_b / idx

    avg_psnr_y = avg_psnr_y / idx
    avg_ssim_y = avg_ssim_y / idx

    # testing log
    logger.info(
        'Average PSNR : {:.4f} , SSIM : {:.4f} , PSNR_b : {:.4f} , PSNR_y : {:.4f} , SSIM_y : {:.4f}\n'.format(avg_psnr, avg_ssim, avg_psnr_b, avg_psnr_y, avg_ssim_y))


if __name__ == '__main__':
    main()
