import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
import matplotlib as matplotlib
import imageio
import logging
from torch.jit._builtins import math
from skimage import measure
import trimesh
from . import grid

def get_root_logger(log_level=logging.INFO, handlers=()):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

def file_backup(backup_dir):
    dir_lis = self.conf['general.recording']
    os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
    for dir_name in dir_lis:
        cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir(dir_name)
        for f_name in files:
            if f_name[-3:] == '.py':
                copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

    copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


''' Extend Adam to support per-voxel learning rate
'''
class Adam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.per_lr = None
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]['params'][0].shape == count.shape
        self.per_lr = count.float() / count.max()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            per_lrs = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)
                    if self.per_lr is not None and p.shape == self.per_lr.shape:
                        per_lrs.append(self.per_lr)
                    else:
                        per_lrs.append(None)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'],
                 per_lrs=per_lrs)
        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         per_lrs):

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        per_lr = per_lrs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if per_lr is not None:
            param.addcdiv_(exp_avg * per_lr, denom, value=-step_size)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)


def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'name':k})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return Adam(param_group, betas=(0.9,0.99))


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer, strict=True):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    if model.rgbnet[0].weight.shape != ckpt['model_state_dict']['rgbnet.0.weight'].shape:
        tmp_weight = torch.zeros(model.rgbnet[0].weight.shape)
        h = ckpt['model_state_dict']['rgbnet.0.weight'].shape[-1]
        tmp_weight[:,:h] = ckpt['model_state_dict']['rgbnet.0.weight']
        ckpt['model_state_dict']['rgbnet.0.weight'] = tmp_weight
    model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    if not no_reload_optimizer:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except:
            print("Failed to load optimizer state dict")
            if strict:
                raise ValueError
            else:
                print("Skip!")
    return model, optimizer, start

def load_grid_data(model, ckpt_path, deduce=1, name='density', return_raw=False):
    ckpt = torch.load(ckpt_path)
    module = getattr(model, name)
    print(">>> {} loaded from ".format(name), ckpt_path)
    if name not in ckpt['model_state_dict']:
        name = name + '.grid'
    if return_raw:
        return ckpt['model_state_dict'][name]
    else:
        if isinstance(module, grid.DenseGrid):
            module.grid.data = ckpt['model_state_dict'][name]
        else:
            module.data = ckpt['model_state_dict'][name]
        return model

def load_weight_by_name(model, ckpt_path, deduce=1, name='density', return_raw=False):
    ckpt = torch.load(ckpt_path)
    for n, module in model.named_parameters():
        if name in n:
            if n in ckpt['model_state_dict']:
                module.data = ckpt['model_state_dict'][n]
                print('load {} to model'.format(n))
    print(">>> data with name {} are loaded from ".format(name), ckpt_path)
    return model

def load_model(model_class, ckpt_path, new_kwargs=None, strict=False):
    ckpt = torch.load(ckpt_path)
    if new_kwargs is not None:
        for k, v in new_kwargs.items():
            if k in ckpt['model_kwargs']:
                if ckpt['model_kwargs'][k] != v:
                    print('updating {} from {} to {}'.format(k, ckpt['model_kwargs'][k], v))
        ckpt['model_kwargs'].update(new_kwargs)

    model = model_class(**ckpt['model_kwargs'])
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print(">>> Checkpoint loaded successfully from {}".format(ckpt_path))
    except Exception as e:
        print(e)
        if strict:
            print(">>> Failed to load checkpoint correctly.")
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        else:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(">>> Checkpoint loaded without strict matching from {}".format(ckpt_path))
    return model

def color_map_color(value, cmap_name='coolwarm', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:,:3]  # will return rgba, we take only first 3 so we get rgb
    return rgb

def analyze_point_cloud(filename=None, log_num=18, rand_offset=False,
                        query_func=None, scale_mats_np=None, save_root=''):
    plydata = PlyData.read(filename)
    num_points = 2 ** log_num
    skip = len(plydata['vertex']) // num_points
    idx = np.arange(len(plydata['vertex']))[::skip]
    if rand_offset:
        rand = np.random.randint(skip)
        idx[:-1] += rand
    points = np.vstack([[v[0],v[1],v[2]] for v in plydata['vertex'][idx]])
    if query_func is None:
        return points
    if scale_mats_np is not None:
        point_ = (points - scale_mats_np[:3,3]) / scale_mats_np[0,0]
    else:
        point_ = points
    batch_size = 8192
    sdfs = []
    for i in range(int(np.ceil(len(points) / batch_size))):
        pts = torch.from_numpy(point_[i*batch_size : (i+1)*batch_size]).cuda()
        sdf = -query_func(pts)
        sdfs.append(sdf.cpu().numpy())
    sdfs = np.hstack(sdfs)
    colors = (color_map_color(sdfs * 0.5 + 0.5) * 255).astype(np.uint8)

    vertexs = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(os.path.join(save_root, "gt_pcd_eval.ply"))
    print(">>> Points saved at {}".format(os.path.join(save_root, "gt_pcd_eval.ply")))
    return

def load_point_cloud(filename=None, log_num=17, rand_offset=False, load_normal=False, save_root=''):
    plydata = PlyData.read(filename)

    num_points = 2 ** log_num
    if log_num > 0:
        skip = len(plydata['vertex']) // num_points
    else:
        skip = 1
    idx = np.arange(len(plydata['vertex']))[::skip]

    if rand_offset:
        rand = np.random.randint(skip)
        idx[:-1] += rand
    points = np.vstack([[v[0],v[1],v[2]] for v in plydata['vertex'][idx]])
    if load_normal:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        radius = 3
        # import ipdb; ipdb.set_trace()
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                              max_nn=30))
        normals = np.asarray(pcd.normals)
        normals[:,2], normals[:,2] = normals[:,1], normals[:,2]
    else:
        normals = points / np.linalg.norm(points, 2, -1, True)
    colors = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)

    vertexs = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]
    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(os.path.join(save_root, "est_normal.ply"))
    print(">>> Points saved at {}".format(os.path.join(save_root, "est_normal.ply")))
    exit()
    return

# def write_ply(points, normals=None, colors=None, save_root=''):
#     # from plyfile import PlyData, PlyElement
#     vertexs = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#     if colors is not None:
#         vertex_colors = np.array([tuple(v) for v in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
#         vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
#         for prop in vertex_colors.dtype.names:
#             vertex_all[prop] = vertex_colors[prop]
#     else:
#         vertex_all = np.empty(len(vertexs), vertexs.dtype.descr)
#     for prop in vertexs.dtype.names:
#         vertex_all[prop] = vertexs[prop]
#     el = PlyElement.describe(vertex_all, 'vertex')
#     PlyData([el]).write(os.path.join(save_root, "tmp.ply"))

def write_ply(points, filename, colors=None, normals=None):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.array([tuple(n) for n in normals], dtype=[('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.array([tuple(c * 255) for c in colors],
                                dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)
    # if not os.path.exists(os.path.dirname(filename)):
    #     os.makedirs(os.path.dirname(filename))
    ply.write(filename)

def point_cloud_from_rays(ray_pts, weights, normals):
    import ipdb; ipdb.set_trace()


''' color space process methods
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradLayer(nn.Module):

    def __init__(self, ksize=3):
        super(GradLayer, self).__init__()

        self.ksize = ksize
        if ksize == 0:
            kernel_v = np.asarray(
                [[-1, 0],
                 [ 1, 0]])
            kernel_h = np.asarray(
                [[-1, 1],
                 [ 0, 0]])
        elif ksize == 1:
            kernel_v = np.asarray(
                [[0, -1, 0],
                 [0, 0, 0],
                 [0, 1, 0]])
            kernel_h = np.asarray(
                [[0, 0, 0],
                 [-1, 0, 1],
                 [0, 0, 0]])
            self.ksize = 3
        elif ksize == 3:
            # kernel_v = np.asarray(
            #     [[0, -1, 0],
            #      [0, 0, 0],
            #      [0, 1, 0]])
            # kernel_h = np.asarray(
            #     [[0, 0, 0],
            #      [-1, 0, 1],
            #      [0, 0, 0]])
            # sobel
            kernel_v = np.asarray(
                [[-1,-2,-1],
                 [0,  0, 0],
                 [1,  2, 1]])
            kernel_h = np.asarray(
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
        elif ksize == 5:
            kernel_v = np.asarray(
                [[-1, -4,  -6, -4, -1],
                 [-2, -8, -12, -8, -2],
                 [ 0,  0,   0,  0,  0],
                 [ 2,  8,  12,  8,  2],
                 [ 1,  4,   6,  4,  1],
                 ])
            kernel_h = kernel_v.T
        else:
            raise NotImplementedError

        kernel_v = torch.FloatTensor(kernel_v/np.abs(kernel_v).sum()).unsqueeze(0).unsqueeze(0)
        kernel_h = torch.FloatTensor(kernel_h/np.abs(kernel_h).sum()).unsqueeze(0).unsqueeze(0)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):

        if x.shape[1] == 3:
            x = self.get_gray(x)
        if self.ksize == 0:
            x_v = torch.zeros_like(x)
            x_h = torch.zeros_like(x)
            x_v[...,1:,:] = (x[...,1:,:] - x[...,:-1,:]) / 2
            x_h[...,1:] = (x[...,1:] - x[...,:-1]) / 2
        else:
            x_v = F.conv2d(x, self.weight_v, padding=self.ksize//2)
            x_h = F.conv2d(x, self.weight_h, padding=self.ksize//2)
        # x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        x = torch.cat([x_v, x_h],1)

        return x

class GaussianLayer(nn.Module):

    def __init__(self, ksize=3):
        super(GaussianLayer, self).__init__()
        self.ksize = ksize
        if ksize == 3:
            kernel = np.asarray(
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]])
        elif ksize == 5:
            kernel = np.asarray(
                [[1,  4,  7,  4, 1],
                 [4, 16, 26, 16, 4],
                 [7, 26, 41, 26, 7],
                 [4, 16, 26, 16, 4],
                 [1,  4,  7,  4, 1],
                 ])
        else:
            raise NotImplementedError

        kernel = torch.FloatTensor(kernel/np.abs(kernel).sum()).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):

        x = F.conv2d(x, self.weight, padding=self.ksize//2)

        return x

def _gaussian_3dconv(ksize=3, sigma=1):
    x = np.arange(-(ksize//2),ksize//2 + 1,1)
    y = np.arange(-(ksize//2),ksize//2 + 1,1)
    z = np.arange(-(ksize//2),ksize//2 + 1,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel = torch.from_numpy(kernel).cuda()
    m = nn.Conv3d(1,1,ksize,stride=1,padding=ksize//2, padding_mode='replicate')
    m.weight.data = kernel[None, None, ...] / kernel.sum()
    m.bias.data = torch.zeros(1)
    for param in m.parameters():
        param.requires_grad = False
    # print(kernel)
    return m

class GradLoss(nn.Module):

    def __init__(self, ksize=3, gaussian=True):
        super(GradLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.grad_layer = GradLayer(ksize=ksize)
        self.gaussian = gaussian
        if self.gaussian:
            self.gaussian_layer = GaussianLayer(ksize=3)

    def forward(self, output, gt_img, savedir=''):
        if self.gaussian:
            # output = self.gaussian_layer(output)
            gt_img = self.gaussian_layer(gt_img)
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        loss = self.loss(output_grad, gt_grad)
        if savedir:
            img1 = np.concatenate([to8b(gt_img.detach().cpu().numpy())[0,0][...,None],
                                   to8b(5*gt_grad.detach().cpu().numpy())[0,0][...,None],
                                   to8b(5*gt_grad.detach().cpu().numpy())[0,1][...,None]], axis=1)
            img2 = np.concatenate([to8b(output.detach().cpu().numpy())[0,0][...,None],
                                   to8b(5*output_grad.detach().cpu().numpy())[0,0][...,None],
                                   to8b(5*output_grad.detach().cpu().numpy())[0,1][...,None]], axis=1)
            img8 = np.concatenate([img1, img2], axis=0)
            if not os.path.exists(os.path.join(savedir, "debug_figs")):
                os.mkdir(os.path.join(savedir, "debug_figs"))
            imageio.imwrite(os.path.join(savedir, "debug_figs", "grad_module_{}.png".format(loss)), img8)
        return loss


def rgb_to_luminance(rgb, return_chromaticity=False, gamma_correction=False, lum_avg=1):
    # todo: gamma correction?
    luminance = 0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]
    luminance = luminance / lum_avg
    if return_chromaticity:
        chromaticity = rgb / (luminance[..., None] + 1e-5)
        return luminance[..., None], chromaticity
    return luminance[..., None]

def get_sobel(img, ksize=3, thrd=0.1, g_ksize=0, d_ksize=0, suffix='', vis=False):
    if img.shape[-1] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_sobel_x = cv2.Sobel(img, -1, 1, 0, ksize=ksize)
    img_sobel_y = cv2.Sobel(img, -1, 0, 1, ksize=ksize)
    # img_sobel_xy = cv2.Sobel(img, -1, 1, 1, ksize=ksize)
    absx = cv2.convertScaleAbs(img_sobel_x)
    absy = cv2.convertScaleAbs(img_sobel_y)
    sobel = cv2.addWeighted(absx, 0.5, absy, 0.5,0)

    if g_ksize > 0:
        gaussian = cv2.GaussianBlur(sobel,(g_ksize,g_ksize),0)
    else:
        gaussian = sobel

    if vis:
        titles = ['Original', 'Sobel','gaussian']
        images = [img, sobel, gaussian]
        for i in range(len(titles)):
            plt.subplot(1, len(titles), i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.savefig("debug_figs/test_sobel_k{}_{}.jpg".format(ksize, suffix), dpi=200)
    return gaussian

def calc_grad(img, delta=1, padding=None, kernel=None):
    # image: (H, W, C)
    grad_x = (img[delta:] - img[:-delta]) / delta
    grad_y = (img[: ,delta:] - img[:, :-delta]) / delta
    mid = delta // 2 + 1
    if padding is None:
        grad = torch.cat([grad_x[:, delta:], grad_y[delta:, :]], -1)
    else:
        raise NotImplementedError
    return grad

''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

"""
Sampling strategies
"""
def up_sample(rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
    """
    Up sampling give a fixed inv_s
    copied from neus
    """
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
    inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
    return z_samples

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


"""
Ref-NeRF utils
"""

def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2**torch.arange(min_deg, max_deg)
    shape = x.shape[:-1] + (int(x.shape[1] * scales.shape[0]),)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(shape)
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], axis=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], axis=-1)
    else:
        return four_feat

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.
    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).
    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).
    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.
    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.
    Args:
      deg_view: number of spherical harmonics degrees to use.
    Returns:
      A function for evaluating integrated directional encoding.
    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)
    ml_array = torch.from_numpy(ml_array).cuda()
    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).
        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.
        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(mat.shape[0])], axis=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat([(x + 1j * y)**m for m in ml_array[0, :]], axis=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.cat([torch.real(ide), torch.imag(ide)], axis=-1)

    return integrated_dir_enc_fn

def generate_enc_fn(mode, deg_view):
    if mode == 'pos_enc':
        def dir_enc_fn(direction, _):
            return pos_enc(
                direction, min_deg=0, max_deg=deg_view, append_identity=True)
        return dir_enc_fn, 3 + 3 * deg_view * 2
    elif mode == 'ide':
        ide_dims = [4, 10, 20, 38]
        return generate_ide_fn(deg_view), ide_dims[deg_view-1]
    else:
        raise NameError
# def generate_dir_enc_fn(deg_view):
#     """Generate directional encoding (DE) function.
#     Args:
#       deg_view: number of spherical harmonics degrees to use.
#     Returns:
#       A function for evaluating directional encoding.
#     """
#     integrated_dir_enc_fn = generate_ide_fn(deg_view)
#
#     def dir_enc_fn(xyz):
#         """Function returning directional encoding (DE)."""
#         return integrated_dir_enc_fn(xyz, torch.zeros_like(xyz[..., :1]))
#
#     return dir_enc_fn

@torch.no_grad()
def get_surface_sliding(sdf, resolution=512, grid_boundary=[-1.1, 1.1], level=0):
    avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
    upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    assert resolution % 512 == 0
    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = [grid_boundary[0], grid_boundary[0], grid_boundary[0]]
    grid_max = [grid_boundary[1], grid_boundary[1], grid_boundary[1]]
    xs = np.linspace(grid_min[0], grid_max[0], N+1)
    ys = np.linspace(grid_min[1], grid_max[1], N+1)
    zs = np.linspace(grid_min[2], grid_max[2], N+1)

    print(xs)
    print(ys)
    print(zs)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i+1]
                y_min, y_max = ys[j], ys[j+1]
                z_min, z_max = zs[k], zs[k+1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
                
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z
            
                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):            
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]
                
                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min)/cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()
                    
                    if mask is None:    
                        pts_sdf = evaluate(pts)
                    else:                    
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        #import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.

                z = pts_sdf.detach().cpu().numpy()

                if (not (np.min(z) > level or np.max(z) < level)):
                    z = z.astype(np.float32)
                    verts, faces, normals, values = measure.marching_cubes(
                    volume=z.reshape(cropN, cropN, cropN), #.transpose([1, 0, 2]),
                    level=level,
                    spacing=(
                            (x_max - x_min)/(cropN-1),
                            (y_max - y_min)/(cropN-1),
                            (z_max - z_min)/(cropN-1) ))
                    print(np.array([x_min, y_min, z_min]))
                    print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    print(verts.min(), verts.max())
                    
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    #meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)

    return combined



# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based', ema_scale_shift=False, momentum=0.9, detach_scale_shift=False):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

        self.ema_scale_shift = ema_scale_shift
        self.detach_scale_shift = detach_scale_shift
        self.momentum = momentum
        if self.ema_scale_shift:
            self.register_buffer('scale', torch.tensor([0]).float())
            self.register_buffer('shift', torch.tensor([0]).float())

    def forward(self, prediction, target, mask, share_scale_shift=False):

        if share_scale_shift:
            prediction_ = prediction.view(1, -1, prediction.size(-1))
            target_ = target.view(1, -1, target.size(-1))
            mask_ = mask.view(1, -1, mask.size(-1))
            scale_, shift_ = compute_scale_and_shift(prediction_, target_, mask_)
            if self.detach_scale_shift:
                scale_ = scale_.detach()
                shift_ = shift_.detach()
            if self.ema_scale_shift:
                if self.scale.item() == 0:
                    self.scale.data = scale_
                if self.shift.item() == 0:
                    self.shift.data = shift_
                self.scale.data = self.momentum * self.scale.data + (1 - self.momentum) * scale_
                self.shift.data = self.momentum * self.shift.data + (1 - self.momentum) * shift_
                scale = self.scale.expand(prediction.size(0))
                shift = self.shift.expand(prediction.size(0))
            else:
                scale = scale_.expand(prediction.size(0))
                shift = shift_.expand(prediction.size(0))
        else:
            scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy