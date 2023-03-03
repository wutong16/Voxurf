import numpy as np
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import cv2
# import argparse
# from glob import glob
import torch
import os
import argparse
import glob
import imageio


def _load_colmap(basedir, convert=True, suffix=''):

    poses_arr = np.load(os.path.join(basedir, 'poses_bounds{}.npy'.format(suffix)))
    poses_arr = poses_arr[:, :15].reshape([-1, 3, 5]) # N x 3 x 5
    if convert:
        poses = poses_arr.transpose(1,2,0)
        # from llff to opencv
        poses = np.concatenate([poses[:, 1:2, :],
                                poses[:, 0:1, :],
                                -poses[:, 2:3, :],
                                poses[:, 3:4, :],
                                poses[:, 4:5, :]], 1)
        poses_arr = poses.transpose(2,0,1)
    poses = poses_arr[:,:,:4]
    hwf = poses_arr[0,:3,-1]
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    R = poses[:, :3, :3].transpose(0, 2, 1)  # (B, 3, 3)
    t = -torch.bmm(torch.from_numpy(R), torch.from_numpy(poses[:, :3, 3:])).numpy()  # (B, 3, 1)
    bottom = np.repeat(np.array([0,0,0,1.]).reshape([1, 1, 4]), R.shape[0], axis=0)
    w2c0 = np.concatenate([R, t], -1)
    # w2c = np.concatenate([w2c0, bottom], 1)
    # P = np.matmul(K, w2c)

    Ks = np.repeat(K[None,...], R.shape[0], axis=0)
    P0 = torch.bmm(torch.from_numpy(Ks), torch.from_numpy(w2c0)).numpy()
    P = np.concatenate([P0, bottom], 1)
    # from opencv to opengl
    # aa = np.linalg.inv(w2c) # the same as poses

    camera_dict = {'world_mat_%d' % idx: P[idx] for idx in range(len(P))}
    np.savez(os.path.join(basedir, 'cameras.npz'), **camera_dict)

def blendedmvs_to_NeuS(basedir):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))
    print(pose_paths)
    import ipdb; ipdb.set_trace()
    all_poses = []
    all_imgs = []
    i_split = [[], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    i_split.append(i_split[-1])

    path_intrinsics = os.path.join(basedir, 'intrinsics.txt')
    H, W = imgs[0].shape[:2]
    K = np.loadtxt(path_intrinsics)
    focal = float(K[0,0])

    R = poses[:, :3, :3].transpose(0, 2, 1)  # (B, 3, 3)
    t = -torch.bmm(torch.from_numpy(R), torch.from_numpy(poses[:, :3, 3:])).numpy()  # (B, 3, 1)
    bottom = np.repeat(np.array([0,0,0,1.]).reshape([1, 1, 4]), R.shape[0], axis=0)
    w2c0 = np.concatenate([R, t], -1)

    Ks = np.repeat(K[None,:3,:3], R.shape[0], axis=0)

    P0 = torch.bmm(torch.from_numpy(Ks).float(), torch.from_numpy(w2c0).float()).numpy()
    P = np.concatenate([P0, bottom], 1)
    camera_dict = {'world_mat_%d' % idx: P[idx] for idx in range(len(P))}
    np.savez(os.path.join(basedir, 'cameras.npz'), **camera_dict)

    print("done")

def MVS_to_NeuS(basedir, cam_dir='cams'):
    camera_files = os.listdir(os.path.join(basedir,cam_dir))
    poses = []
    K = None
    for i in range(len(camera_files)):
        file = "{:>08d}_cam.txt".format(i)
        camera_file = os.path.join(basedir, cam_dir, file)
        intrinsics, extrinsics, depth_params = read_cam_file(camera_file)
        poses.append(extrinsics[None,...])
        K = intrinsics
    poses = np.vstack(poses)
    # MVS extrinsic is world2cam already
    # R = poses[:,:3,:3].transpose(0, 2, 1)
    # t = -torch.bmm(torch.from_numpy(R), torch.from_numpy(poses[:, :3, 3:])).numpy()

    R = poses[:,:3,:3]
    t = poses[:,:3,3:]
    bottom = np.repeat(np.array([0,0,0,1.]).reshape([1, 1, 4]), R.shape[0], axis=0)
    w2c0 = np.concatenate([R, t], -1)
    Ks = np.repeat(K[None,...], R.shape[0], axis=0)
    P0 = torch.bmm(torch.from_numpy(Ks), torch.from_numpy(w2c0)).numpy()
    P = np.concatenate([P0, bottom], 1)
    camera_dict = {'world_mat_%d' % idx: P[idx] for idx in range(len(P))}
    np.savez(os.path.join(basedir, 'cameras.npz'), **camera_dict)
    print('done, saved at', os.path.join(basedir, 'cameras.npz'))

def TAT0_to_NeuS(basedir, cam_dir='pose'):
    poses = []
    K = None
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    for i, file in enumerate(pose_paths):
        pose = np.loadtxt(file).astype(np.float32)
        # intrinsics, extrinsics, depth_params = read_cam_file(camera_file)
        poses.append(np.linalg.inv(pose)[None,...])
    poses = np.vstack(poses).astype(np.float32)
    path_intrinsics = os.path.join(basedir, 'intrinsics.txt')
    K = np.loadtxt(path_intrinsics)[:3,:3].astype(np.float32)
    # MVS extrinsic is world2cam already
    # R = poses[:,:3,:3].transpose(0, 2, 1)
    # t = -torch.bmm(torch.from_numpy(R), torch.from_numpy(poses[:, :3, 3:])).numpy()

    R = poses[:,:3,:3]
    t = poses[:,:3,3:]
    bottom = np.repeat(np.array([0,0,0,1.]).reshape([1, 1, 4]), R.shape[0], axis=0)
    w2c0 = np.concatenate([R, t], -1)
    Ks = np.repeat(K[None,...], R.shape[0], axis=0)
    P0 = torch.bmm(torch.from_numpy(Ks), torch.from_numpy(w2c0)).numpy()
    P = np.concatenate([P0, bottom], 1)
    camera_dict = {'world_mat_%d' % idx: P[idx] for idx in range(len(P))}
    np.savez(os.path.join(basedir, 'cameras.npz'), **camera_dict)
    print('done, saved at', os.path.join(basedir, 'cameras.npz'))

def NeuS_to_MVS(basedir):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    bds = poses_arr[:, -2:].transpose([1,0])
    near, far = bds[bds[:,0] > 0, 0].min() * 0.8, bds[bds[:,1] > 0, 1].max() * 1.2
    poses_arr = poses_arr[:, :15].reshape([-1, 3, 5]) # N x 3 x 5

    poses = poses_arr.transpose(1,2,0)
    # from llff to opencv
    poses = np.concatenate([poses[:, 1:2, :],
                            poses[:, 0:1, :],
                            -poses[:, 2:3, :],
                            poses[:, 3:4, :],
                            poses[:, 4:5, :]], 1)
    poses_arr = poses.transpose(2,0,1)

    # camera to world
    poses = poses_arr[:,:,:4]

    hwf = poses_arr[0,:3,-1]
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    R = poses[:, :3, :3].transpose(0, 2, 1)  # (B, 3, 3)
    t = -torch.bmm(torch.from_numpy(R), torch.from_numpy(poses[:, :3, 3:])).numpy()  # (B, 3, 1)
    bottom = np.repeat(np.array([0,0,0,1.]).reshape([1, 1, 4]), R.shape[0], axis=0)
    w2c0 = np.concatenate([R, t], -1)
    P = np.concatenate([w2c0, bottom], 1)

    intrinsics, extrinsics = K, P
    if not os.path.exists(os.path.join(basedir, 'cams_1')):
        os.mkdir(os.path.join(basedir, 'cams_1'))
    for i in range(poses.shape[0]):
        file = "{:>08d}_cam.txt".format(i)
        camera_file = os.path.join(basedir, 'cams_1', file)
        with open(camera_file, "w") as f:
            f.write("extrinsic\n")
            for l in extrinsics[i]:
                seq = ["{:.6f} ".format(e) for e in l] + ['\n']
                f.writelines( seq )
            f.write("\nintrinsic\n")
            for l in intrinsics:
                seq = ["{:.6f} ".format(e) for e in l] + ['\n']
                f.writelines(seq)
            f.write("\n{:.2f} {:.2f}\n".format(near, far))


def read_cam_file(filename):
    """Read camera intrinsics, extrinsics, and depth values (min, max) from text file

    Args:
        filename: cam text file path string

    Returns:
        Tuple with intrinsics matrix (3x3), extrinsics matrix (4x4), and depth params vector (min and max) if exists
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # depth min and max: line 11
    if len(lines) >= 12:
        depth_params = np.fromstring(lines[11], dtype=np.float32, sep=' ')
    else:
        depth_params = np.empty(0)

    return intrinsics, extrinsics, depth_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='', help='data source folder for preprocess')
    parser.add_argument('--mode', type=str, default='colmap', help='what kind of source format to convert')
    opt = parser.parse_args()

    if opt.mode == 'colmap':
        _load_colmap(opt.source_dir, True)
    elif opt.mode == 'mvs2neus':
        MVS_to_NeuS(opt.source_dir)
    elif opt.mode == 'tat02neus':
        TAT0_to_NeuS(opt.source_dir)
    elif opt.mode == 'neus2mvs':
        NeuS_to_MVS(opt.source_dir)
    elif opt.mode == 'blendedmvs2neus':
        blendedmvs_to_NeuS(opt.source_dir)
    else:
        raise NotImplementedError

