import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_dtu_data(basedir, normalize=True, reso_level=2, mask=True, white_bg=True):

    rgb_paths = sorted(glob(os.path.join(basedir, 'image', '*png')))
    if len(rgb_paths) == 0:
        rgb_paths = sorted(glob(os.path.join(basedir, 'image', '*jpg')))
    if len(rgb_paths) == 0:
        rgb_paths = sorted(glob(os.path.join(basedir, 'rgb', '*png')))

    mask_paths = sorted(glob(os.path.join(basedir, 'mask', '*png')))
    if len(mask_paths) == 0:
        mask_paths = sorted(glob(os.path.join(basedir, 'mask', '*jpg')))

    render_cameras_name = 'cameras_sphere.npz' if normalize else 'cameras_large.npz'
    camera_dict = np.load(os.path.join(basedir, render_cameras_name))
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(rgb_paths))]
    if normalize:
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(rgb_paths))]
    else:
        scale_mats_np = None
    all_intrinsics = []
    all_poses = []
    all_imgs = []
    all_masks = []
    for i, (world_mat, im_name) in enumerate(zip(world_mats_np, rgb_paths)):
        if normalize:
            P = world_mat @ scale_mats_np[i]
        else:
            P = world_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        all_intrinsics.append(intrinsics)
        all_poses.append(pose)
        if len(mask_paths) > 0:
            mask_ = (imageio.imread(mask_paths[i]) / 255.).astype(np.float32)
            if mask_.ndim == 3:
                all_masks.append(mask_[...,:3])
            else:
                all_masks.append(mask_[...,None])
        all_imgs.append((imageio.imread(im_name) / 255.).astype(np.float32))
    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    H, W = imgs[0].shape[:2]
    K = all_intrinsics[0]
    focal = all_intrinsics[0][0,0]
    print("Date original shape: ", H, W)
    masks = np.stack(all_masks, 0)
    if mask:
        assert len(mask_paths) > 0
        bg = 1. if white_bg else 0.
        imgs = imgs * masks + bg * (1 - masks)

    if reso_level > 1:
        H, W = int(H / reso_level), int(W / reso_level)
        imgs =  F.interpolate(torch.from_numpy(imgs).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        if masks is not None:
            masks =  F.interpolate(torch.from_numpy(masks).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        K[:2] /= reso_level
        focal /= reso_level

    # this is to randomly fetch images.
    i_test = [8, 13, 16, 21, 26, 31, 34]
    if len(imgs) * 0.1 >= 8:
        print("add 56 to test set")
        i_test.append(56)
    i_test = [i for i in i_test if i < len(imgs)]
    i_val = i_test
    i_train = list(set(np.arange(len(imgs))) - set(i_test))

    i_split = [np.array(i_train), np.array(i_val), np.array(i_test)]

    render_poses = poses[i_split[-1]]
    return imgs, poses, render_poses, [H, W, focal], K, i_split, scale_mats_np[0], masks

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)