# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58,    0, 0.5, 0],
                           [   0, 1.92, 0.5, 0],
                           [   0,    0,   1, 0],
                           [   0,    0,   0, 1]], dtype=np.float32)

        self.full_res_shape = (344, 192)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, do_flip):
        color = self.loader(self.get_image_path(folder))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder):
        image_path = os.path.join(
            self.data_path, folder)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder):
        image_path = os.path.join(
            self.data_path, folder)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder ):
        image_path = os.path.join(
            self.data_path, folder)
        return image_path

    def get_depth(self, folder, do_flip):
        # f_str = "{:010d}.png".format(frame_index)
        root = "/media/thc/Elements/kittidepth/train"
        # file = root + folder[10:]
        # depth_path = os.path.join(

        #     file,
        #     "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
        #     f_str)

        depth_path = os.path.join(root, folder)
        depth_gt = pil.open(depth_path)
        # depth_gt = depth_gt[0:800,:]
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    # def get_depth(self, folder, do_flip, frame_index):
    #     calib_path = os.path.join(self.data_path, folder.split("/")[0])

    #     velo_filename = os.path.join(
    #         self.data_path,
    #         folder,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    #     depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map["l"])
    #     depth_gt = skimage.transform.resize(
    #         depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt
