import os
import torch
import torch.nn.functional as F
import numpy as np

from glob import glob
import cv2
import random
import imageio
import skimage


def load_rgb(path, normalize_rgb = False):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # if normalize_rgb: # [-1,1] --> [0,1]
    #     img -= 0.5
    #     img *= 2.
    return img


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def load_scannet_data(data_dir, img_res=[384, 384], center_crop_type='no_crop', use_mask=False, num_views=-1):

    # instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))
    instance_dir = data_dir

    total_pixels = img_res[0] * img_res[1]
    img_res = img_res
    num_views = num_views
    assert num_views in [-1, 3, 6, 9]

    assert os.path.exists(instance_dir), "Data directory is empty"

    image_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_rgb.png"))
    depth_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_depth.npy"))
    normal_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_normal.npy"))
    
    # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
    if use_mask:
        mask_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_mask.npy"))
    else:
        mask_paths = None

    n_images = len(image_paths)
    
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]


    # cam_file_2 = '{0}/cameras_sphere.npz'.format(instance_dir)
    # camera_dict_2 = np.load(cam_file_2)
    # scale_mats_2 = [camera_dict_2['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    # world_mats_2 = [camera_dict_2['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    # for i in range(n_images):
    #     assert np.sum(np.abs(scale_mats[i] - scale_mats_2[i])) == 0
    #     assert np.sum(np.abs(world_mats[i] - world_mats_2[i])) == 0


    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)

        # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
        if center_crop_type == 'center_crop_for_replica':
            scale = 384 / 680
            offset = (1200 - 680 ) * 0.5
            intrinsics[0, 2] -= offset
            intrinsics[:2, :] *= scale
        elif center_crop_type == 'center_crop_for_tnt':
            scale = 384 / 540
            offset = (960 - 540) * 0.5
            intrinsics[0, 2] -= offset
            intrinsics[:2, :] *= scale
        elif center_crop_type == 'center_crop_for_dtu':
            scale = 384 / 1200
            offset = (1600 - 1200) * 0.5
            intrinsics[0, 2] -= offset
            intrinsics[:2, :] *= scale
        elif center_crop_type == 'padded_for_dtu':
            scale = 384 / 1200
            offset = 0
            intrinsics[0, 2] -= offset
            intrinsics[:2, :] *= scale
        elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
            pass
        else:
            raise NotImplementedError
        
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)

    rgb_images = []
    for path in image_paths:
        rgb = load_rgb(path)
        rgb_images.append(rgb)

    imgs = np.stack(rgb_images, 0)
    poses = np.stack(pose_all, 0)
    K = np.stack(intrinsics_all, 0)
    K = intrinsics_all[0]

    H, W = imgs[0].shape[:2]
    focal = intrinsics_all[0][0,0]

        
    depth_images = []
    normal_images = []

    for dpath, npath in zip(depth_paths, normal_paths):
        depth = np.load(dpath)
        depth_images.append(depth)
    
        normal = np.load(npath)
        # important as the output of omnidata is normalized
        normal = normal * 2. - 1.
        normal = np.transpose(normal, (1,2,0))
        normal_images.append(normal)

    depth_images = np.stack(depth_images, 0)
    normal_images = np.stack(normal_images, 0)
    # load mask
    mask_images = []
    if mask_paths is None:
        for rgb in rgb_images:
            mask = np.ones_like(rgb[:, :, :1])
            mask_images.append(mask)
    else:
        for path in mask_paths:
            mask = np.load(path)
            mask_images.append(mask)

    masks = np.stack(mask_images, 0)
    
    i_split = [np.array(np.arange(len(imgs))), np.array(np.arange(0, len(imgs), 10)), np.array(np.arange(0, len(imgs), 10))]

    render_poses = poses[i_split[-1]]

    return imgs, poses, render_poses, [H, W, focal], K, i_split, scale_mats[0], masks, depth_images, normal_images

    

# if __name__ == "__main__":
#     load_scannet_data('/mnt/petrelfs/wangjiaqi/VoxurF-new/data/scannet/scan1/')

