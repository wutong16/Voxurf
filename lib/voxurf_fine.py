import os
import time
import numpy as np
from copy import deepcopy
import cv2
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.dvgo_ori import extract_geometry
from torch_scatter import segment_coo

from . import grid
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

'''Model'''
class Voxurf(torch.nn.Module):
    """
    This module is modified from DirectVoxGO https://github.com/sunset1995/DirectVoxGO/blob/main/lib/dvgo.py
    """
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4, 
                 center_sdf=False, grad_feat=(1.0,), sdf_feat=(),
                 use_layer_norm=False,
                 grad_mode='interpolate',
                 s_ratio=2000, s_start=0.2, s_learn=False, step_start=0,
                 smooth_sdf=False, 
                 smooth_ksize=0, smooth_sigma=1,
                 k_rgbnet_depth=3, k_res=False, k_posbase_pe=5, k_viewbase_pe=4,
                 k_center_sdf=False, k_grad_feat=(1.0,), k_sdf_feat=(),
                 smooth_scale=True, use_grad_norm=True,
                 use_rgb_k=True, k_detach_1=True, k_detach_2=True,
                 use_rgbnet_k0=False,
                 **kwargs):
        super(Voxurf, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        self.smooth_scale = smooth_scale

        self.s_ratio = s_ratio
        self.s_start = s_start
        self.s_learn = s_learn
        self.step_start = step_start
        self.s_val = nn.Parameter(torch.ones(1), requires_grad=s_learn).cuda()
        self.s_val.data *= s_start
        self.smooth_sdf = smooth_sdf
        self.sdf_init_mode = "ball_init"

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        if self.sdf_init_mode == "ball_init":
            self.sdf = grid.create_grid(
                'DenseGrid', channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            x, y, z = np.mgrid[-1.0:1.0:self.world_size[0].item() * 1j, -1.0:1.0:self.world_size[1].item() * 1j, -1.0:1.0:self.world_size[2].item() * 1j]
            self.sdf.grid.data = torch.from_numpy((x ** 2 + y ** 2 + z ** 2) ** 0.5 - 1).float()[None, None, ...]
        elif self.sdf_init_mode == "random":
            self.sdf = torch.nn.Parameter(torch.rand([1, 1, *self.world_size]) * 0.05) # random initialization
            torch.nn.init.normal_(self.sdf, 0.0, 0.5)
        else:
            raise NotImplementedError

        self.init_smooth_conv(smooth_ksize, smooth_sigma)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe,
        }
        if rgbnet_dim <= 0:
            # color voxel grid  (dvgo coarse stage)
            self.k0_dim = 3
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                            'DenseGrid', channels=self.k0_dim, world_size=self.world_size,
                            xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*posbase_pe*2) + (3+3*viewbase_pe*2)

            self.use_grad_norm = use_grad_norm
            self.center_sdf = center_sdf
            self.grad_feat = grad_feat
            self.sdf_feat = sdf_feat
            self.use_rgb_k = use_rgb_k
            self.k_detach_1 = k_detach_1
            self.k_detach_2 = k_detach_2
            self.use_rgbnet_k0 = use_rgbnet_k0
            self.use_layer_norm = use_layer_norm
            dim0 += len(self.grad_feat) * 3
            dim0 += len(self.sdf_feat) * 6
            if self.use_rgbnet_k0:
                dim0 += self.k0_dim
            if self.center_sdf:
                dim0 += 1
            if not self.use_layer_norm:
                self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
            else:
                self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.LayerNorm(rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.LayerNorm(rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('feature voxel grid', self.k0.grid.shape)
            print('mlp', self.rgbnet)

            # the second rgb net
            self.k_res = k_res
            self.k_center_sdf = k_center_sdf
            self.k_grad_feat = k_grad_feat
            self.k_sdf_feat = k_sdf_feat
            self.register_buffer('k_posfreq', torch.FloatTensor([(2**i) for i in range(k_posbase_pe)]))
            self.register_buffer('k_viewfreq', torch.FloatTensor([(2**i) for i in range(k_viewbase_pe)]))
            k_dim0 = (3+3*k_posbase_pe*2) + (3+3*k_viewbase_pe*2) + self.k0_dim
            if self.k_res:
                k_dim0 += 3
            if self.k_center_sdf:
                k_dim0 += 1
            k_dim0 += len(self.k_grad_feat) * 3
            k_dim0 += len(self.k_sdf_feat) * 6
            if not self.use_layer_norm:
                self.k_rgbnet = nn.Sequential(
                    nn.Linear(k_dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(k_rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
            else:
                self.k_rgbnet = nn.Sequential(
                    nn.Linear(k_dim0, rgbnet_width), nn.LayerNorm(rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.LayerNorm(rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(k_rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('feature voxel grid', self.k0.grid.shape)
            print('k_rgbnet mlp', self.k_rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            self.mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()
        else:
            self.mask_cache = None
            self.nonempty_mask = None

        # grad conv to calculate gradient
        self.init_gradient_conv()
        self.grad_mode = grad_mode

    def init_gradient_conv(self, sigma = 0):
        self.grad_conv = nn.Conv3d(1,3,(3,3,3),stride=(1,1,1), padding=(1, 1, 1), padding_mode='replicate')
        # fixme:  a better operator?
        kernel = np.asarray([
            [[1,2,1],[2,4,2],[1,2,1]],
            [[2,4,2],[4,8,4],[2,4,2]],
            [[1,2,1],[2,4,2],[1,2,1]],
        ])

        # sigma controls the difference between naive [-1,1] and sobel kernel
        distance = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance[i,j,k] = ((i-1)**2 + (j-1)**2 + (k-1)**2 - 1)
        kernel0 = kernel * np.exp(-distance * sigma)

        kernel1 = kernel0 / ( kernel0[0].sum() * 2 * self.voxel_size.item())
        weight = torch.from_numpy(np.concatenate([kernel1[None] for _ in range(3)])).float()
        weight[0,1,:,:] *= 0
        weight[0,0,:,:] *= -1
        weight[1,:,1,:] *= 0
        weight[1,:,0,:] *= -1
        weight[2,:,:,1] *= 0
        weight[2,:,:,0] *= -1

        self.grad_conv.weight.data = weight.unsqueeze(1).float()
        self.grad_conv.bias.data = torch.zeros(3)
        for param in self.grad_conv.parameters():
            param.requires_grad = False

        # smooth conv for TV
        self.tv_smooth_conv = nn.Conv3d(1, 1, (3, 3, 3), stride=1, padding=1, padding_mode='replicate')
        weight = torch.from_numpy(kernel0 / kernel0.sum()).float()
        self.tv_smooth_conv.weight.data = weight.unsqueeze(0).unsqueeze(0).float()
        self.tv_smooth_conv.bias.data = torch.zeros(1)
        for param in self.tv_smooth_conv.parameters():
            param.requires_grad = False
            
        self.mask_kernel =  weight.view(1, -1).float().cuda()

    def _gaussian_3dconv(self, ksize=3, sigma=1):
        x = np.arange(-(ksize//2),ksize//2 + 1,1)
        y = np.arange(-(ksize//2),ksize//2 + 1,1)
        z = np.arange(-(ksize//2),ksize//2 + 1,1)
        xx, yy, zz = np.meshgrid(x,y,z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
        kernel = torch.from_numpy(kernel).to(self.sdf.grid)
        m = nn.Conv3d(1,1,ksize,stride=1,padding=ksize//2, padding_mode='replicate')
        m.weight.data = kernel[None, None, ...] / kernel.sum()
        m.bias.data = torch.zeros(1)
        for param in m.parameters():
            param.requires_grad = False
        return m

    def init_smooth_conv_test_k3(self, ksize=3, sigma=0.4):
        self.smooth_conv_test_k3 = self._gaussian_3dconv(ksize, sigma)
        print("- "*10 + "init smooth conv test with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def init_smooth_conv_test_k5(self, ksize=5, sigma=0.4):
        self.smooth_conv_test_k5 = self._gaussian_3dconv(ksize, sigma)
        print("- "*10 + "init smooth conv test with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def init_smooth_conv(self, ksize=3, sigma=1):
        self.smooth_sdf = ksize > 0
        if self.smooth_sdf:
            self.smooth_conv = self._gaussian_3dconv(ksize, sigma)
            print("- "*10 + "init smooth conv with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def init_feature_smooth_conv(self, ksize=3, sigma=1):
        self.smooth_feature = ksize > 0
        if self.smooth_feature:
            self.feature_smooth_conv = self._gaussian_3dconv(ksize, sigma)
            print("- "*10 + "init feature smooth conv with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def init_sdf_from_sdf(self, sdf0=None, smooth=False, reduce=1., ksize=3, sigma=1., zero2neg=True):
        print("\n", "- "*3 + "initing sdf from sdf" + " -"*3, "\n")
        if sdf0.shape != self.sdf.grid.shape:
            sdf0 = F.interpolate(sdf0, size=tuple(self.world_size), mode='trilinear', align_corners=True)
        if smooth:
            m = self._gaussian_3dconv(ksize, sigma)
            sdf_data = m(sdf0 / reduce)
            self.sdf.grid = torch.nn.Parameter(sdf_data).to(self.sdf.grid) / reduce
        else:
            self.sdf.grid.data = sdf0.to(self.sdf.grid) / reduce # + self.act_shift
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        if self.smooth_scale:
            m = self._gaussian_3dconv(ksize=5, sigma=1)
            with torch.no_grad():
                self.sdf.grid = torch.nn.Parameter(m(self.sdf.grid.data)).cuda()
        self.gradient = self.neus_sdf_gradient()

    def init_sdf_from_density(self, smooth=False, reduce=1., ksize=3, sigma=1., zero2neg=True):
        print("\n", "- "*3 + "initing sdf from density" + " -"*3, "\n")
        self.s = torch.nn.Parameter(torch.ones(1)) * 10
        if zero2neg:
            self.density.data[self.density.data==0] = -100
        if self.density.shape != self.sdf.grid.shape:
            self.density.data = F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True)
        if smooth:
            m = self._gaussian_3dconv(ksize, sigma)
            sdf_data = m(-torch.tanh(self.density.data) / reduce)
            self.sdf.grid = torch.nn.Parameter(sdf_data)
        else:
            self.sdf.grid.data = -torch.tanh(self.density.data) / reduce # + self.act_shift

        self.gradient = self.neus_sdf_gradient()


    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            'grad_feat': self.grad_feat,
            'sdf_feat': self.sdf_feat,
            'k_grad_feat': self.k_grad_feat,
            'k_sdf_feat': self.k_sdf_feat,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.density[~self.nonempty_mask] = -100
        self.sdf.grid[~self.nonempty_mask] = 1


    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100
        self.sdf.grid[nearest_dist[None,None] <= near] = 1

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        self.sdf.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def sdf_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.sdf.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)


    def density_total_variation(self, sdf_tv=0, smooth_grad_tv=0, grad_tv=0, smooth_sdf_tv=0):
        t1 = time.time()
        tv = 0
        if sdf_tv > 0:
            tv += total_variation(self.sdf.grid, self.nonempty_mask) / 2 / self.voxel_size * sdf_tv
        if smooth_grad_tv > 0:
            smooth_tv_error = (self.tv_smooth_conv(self.gradient.permute(1,0,2,3,4)).detach() - self.gradient.permute(1,0,2,3,4))
            smooth_tv_error = smooth_tv_error[self.nonempty_mask.repeat(3,1,1,1,1)] ** 2
            tv += smooth_tv_error.mean() * smooth_grad_tv
        return tv

    def k0_total_variation(self, k0_tv=1., k0_grad_tv=0.):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        tv = 0
        if k0_tv > 0:
            tv += total_variation(v, self.nonempty_mask.repeat(1,v.shape[1],1,1,1))
        if k0_grad_tv > 0:
            raise NotImplementedError
        return tv

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)


    def neus_sdf_gradient(self, mode=None, sdf=None):
        # the gradient grid from the sdf grid
        if sdf is None:
            sdf = self.sdf.grid
        if mode is None:
            mode = self.grad_mode
        if mode == 'interpolate':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:,0,1:-1,:,:] = (sdf[:,0,2:,:,:] - sdf[:,0,:-2,:,:]) / 2 / self.voxel_size
            gradient[:,1,:,1:-1,:] = (sdf[:,0,:,2:,:] - sdf[:,0,:,:-2,:]) / 2 / self.voxel_size
            gradient[:,2,:,:,1:-1] = (sdf[:,0,:,:,2:] - sdf[:,0,:,:,:-2]) / 2 / self.voxel_size
        elif mode == 'grad_conv':
            gradient = self.grad_conv(sdf)
        elif mode == 'raw':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:,0,:-1,:,:] = (sdf[:,0,1:,:,:] - sdf[:,0,:-1,:,:]) / self.voxel_size
            gradient[:,1,:,:-1,:] = (sdf[:,0,:,1:,:] - sdf[:,0,:,:-1,:]) / self.voxel_size
            gradient[:,2,:,:,:-1] = (sdf[:,0,:,:,1:] - sdf[:,0,:,:,:-1]) / self.voxel_size
        else:
            raise NotImplementedError
        return gradient


    def neus_alpha_from_sdf_scatter(self, viewdirs, ray_id, dist, sdf, gradients, global_step,
                            is_train, use_mid=True):
        # force s_val value to change with global step
        if is_train:
            if not self.s_learn:
                s_val = 1. / (global_step + self.s_ratio / self.s_start - self.step_start) * self.s_ratio
                self.s_val.data = torch.ones_like(self.s_val) * s_val
            else:
                s_val = self.s_val.item()
        else:
            s_val = 0

        dirs = viewdirs[ray_id]
        inv_s = torch.ones(1).cuda() / self.s_val
        assert use_mid
        if use_mid:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)
            cos_anneal_ratio = 1.0
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                         F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive (M, 1)

            sdf = sdf.unsqueeze(-1) # (M, 1)

            # dist is a constant in this impelmentation
            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dist.reshape(-1, 1) * 0.5 # (M, 1)
            estimated_prev_sdf = sdf - iter_cos * dist.reshape(-1, 1) * 0.5 # (M, 1)
        else:
            estimated_next_sdf = torch.cat([sdf[..., 1:], sdf[..., -1:]], -1).reshape(-1, 1)
            estimated_prev_sdf = torch.cat([sdf[..., :1], sdf[..., :-1]], -1).reshape(-1, 1)

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s.reshape(-1, 1))
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s.reshape(-1, 1))
        p = prev_cdf - next_cdf

        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).squeeze()
        return s_val, alpha
    
    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True, sample_ret=True, sample_grad=False, displace=0.1, smooth=False):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        
        if smooth:	
            grid = self.smooth_conv(grids[0])
            grids[0] = grid
            
        outs = []
        if sample_ret:
            ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
            grid = grids[0]
            ret = F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(
                grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze(-1)
            outs.append(ret)
        
        if sample_grad:
            grid = grids[0]
            feat, grad = self.sample_sdfs(xyz, grid, displace_list=[1.0], use_grad_norm=False)
            feat = torch.cat([feat[:, 4:6], feat[:, 2:4], feat[:, 0:2]], dim=-1)
            grad = torch.cat([grad[:, [2]], grad[:, [1]], grad[:, [0]]], dim=-1)
            
            outs.append(grad)
            outs.append(feat)
            
        if len(outs) == 1:
            return outs[0]
        else:
            return outs
    
    
    def sample_sdfs(self, xyz, *grids, displace_list, mode='bilinear', align_corners=True, use_grad_norm=False):
        
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        
        grid = grids[0]
        # ind from xyz to zyx !!!!!
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        grid_size = grid.size()[-3:]
        size_factor_zyx = torch.tensor([grid_size[2], grid_size[1], grid_size[0]]).cuda()
        ind = ((ind_norm + 1) / 2) * (size_factor_zyx - 1)
        offset = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]).cuda()
        displace = torch.tensor(displace_list).cuda()
        offset = offset[:, None, :] * displace[None, :, None]
        
        all_ind = ind.unsqueeze(-2) + offset.view(-1, 3)
        all_ind = all_ind.view(1, 1, 1, -1, 3)
        all_ind[..., 0] = all_ind[..., 0].clamp(min=0, max=size_factor_zyx[0] - 1)
        all_ind[..., 1] = all_ind[..., 1].clamp(min=0, max=size_factor_zyx[1] - 1)
        all_ind[..., 2] = all_ind[..., 2].clamp(min=0, max=size_factor_zyx[2] - 1)
        
        all_ind_norm = (all_ind / (size_factor_zyx-1)) * 2 - 1
        feat = F.grid_sample(grid, all_ind_norm, mode=mode, align_corners=align_corners)
        
        all_ind = all_ind.view(1, 1, 1, -1, 6, len(displace_list), 3)
        diff = all_ind[:, :, :, :, 1::2, :, :] - all_ind[:, :, :, :, 0::2, :, :]
        diff, _ = diff.max(dim=-1)
        feat_ = feat.view(1, 1, 1, -1, 6, len(displace_list))
        feat_diff = feat_[:, :, :, :, 1::2, :] - feat_[:, :, :, :, 0::2, :]
        grad = feat_diff / diff / self.voxel_size
        
        feat = feat.view(shape[-1], 6, len(displace_list))
        grad = grad.view(shape[-1], 3, len(displace_list))
        
        if use_grad_norm:
            grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-5)
        
        feat = feat.view(shape[-1], 6 * len(displace_list))
        grad = grad.view(shape[-1], 3 * len(displace_list))
        
        return feat, grad

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        # correct the cuda output N_steps, which could have a bias of 1 randomly
        N_steps = ray_id.unique(return_counts=True)[1]
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, mask_outbbox, N_steps


    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering'''

        ret_dict = {}
        N = len(rays_o)

        ray_pts, ray_id, step_id, mask_outbbox, N_steps = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        # interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        gradient, gradient_error = None, None

        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            mask_outbbox[~mask_outbbox] |= ~mask

        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid

        sdf, gradient, feat = self.grid_sampler(ray_pts, sdf_grid, sample_ret=True, sample_grad=True, displace=1.0)

        dist = render_kwargs['stepsize'] * self.voxel_size
        s_val, alpha = self.neus_alpha_from_sdf_scatter(viewdirs, ray_id, dist, sdf, gradient, global_step=global_step,
                                                 is_train=global_step is not None, use_mid=True)

        mask = None
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            ray_pts = ray_pts[mask]
            step_id = step_id[mask]
            gradient = gradient[mask] # merge to sample once
            sdf = sdf[mask]

        # compute accumulated transmittance
        if ray_id.ndim == 2:
            print(mask, alpha, ray_id)
            mask = mask.squeeze()
            alpha = alpha.squeeze()
            ray_id = ray_id.squeeze()
            ray_pts = ray_pts.squeeze()
            step_id = step_id.squeeze()
            gradient = gradient.squeeze()
            sdf = sdf.squeeze()

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            gradient = gradient[mask]
            sdf = sdf[mask]

        k0 = self.k0(ray_pts)

        all_grad_inds = list(set(self.grad_feat + self.k_grad_feat))
        all_sdf_inds = list(set(self.sdf_feat + self.k_sdf_feat))
        
        assert all_grad_inds == all_sdf_inds
        
        if len(all_grad_inds) > 0: 
            all_grad_inds = sorted(all_grad_inds)
            all_grad_inds_ = deepcopy(all_grad_inds)
            all_feat, all_grad = self.sample_sdfs(ray_pts, sdf_grid, displace_list=all_grad_inds_, use_grad_norm=self.use_grad_norm)
        else:
            all_feat, all_grad = None, None
        
        self.gradient = self.neus_sdf_gradient()
        
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        if self.use_rgbnet_k0:
            rgb_feat = torch.cat([
                k0, xyz_emb, viewdirs_emb.flatten(0, -2)[ray_id]
            ], -1)
        else:
            rgb_feat = torch.cat([
                xyz_emb, viewdirs_emb.flatten(0, -2)[ray_id]
            ], -1)

        hierarchical_feats = []
        if self.center_sdf:
            hierarchical_feats.append(sdf[:, None])
        if len(all_grad_inds) > 0:
            hierarchical_feats.append(all_feat)
            hierarchical_feats.append(all_grad)
        if len(hierarchical_feats) > 0:
            rgb_feat = torch.cat([rgb_feat, *hierarchical_feats], dim=-1)

        rgb_logit = self.rgbnet(rgb_feat)
        rgb = torch.sigmoid(rgb_logit)


        if self.use_rgb_k:
            k_xyz_emb = (rays_xyz.unsqueeze(-1) * self.k_posfreq).flatten(-2)
            k_xyz_emb = torch.cat([rays_xyz, k_xyz_emb.sin(), k_xyz_emb.cos()], -1)
            k_viewdirs_emb = (viewdirs.unsqueeze(-1) * self.k_viewfreq).flatten(-2)
            k_viewdirs_emb = torch.cat([viewdirs, k_viewdirs_emb.sin(), k_viewdirs_emb.cos()], -1)
            k_rgb_feat = torch.cat([
                k0, k_xyz_emb, k_viewdirs_emb.flatten(0, -2)[ray_id]
            ], -1)

            assert len(self.k_grad_feat) == 1 and self.k_grad_feat[0] == 1.0
            assert len(self.k_sdf_feat) == 0
            all_feats_ = [gradient]

            if self.k_center_sdf:
                all_feats_.append(sdf[:, None])
            if len(all_feats_) > 0:
                all_feats_ = torch.cat(all_feats_, dim=-1)
                k_rgb_feat = torch.cat([k_rgb_feat, all_feats_], dim=-1)

            if self.k_res:
                color_feat = rgb_logit
                if self.k_detach_1:
                    k_rgb_feat =  torch.cat([k_rgb_feat, color_feat.detach()], dim=-1)
                else:
                    k_rgb_feat =  torch.cat([k_rgb_feat, color_feat], dim=-1)

            if self.k_detach_2:
                k_rgb_logit = rgb_logit.detach() + self.k_rgbnet(k_rgb_feat)
            else:
                k_rgb_logit = rgb_logit + self.k_rgbnet(k_rgb_feat)
            k_rgb = torch.sigmoid(k_rgb_logit)
            k_rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * k_rgb),
                index=ray_id, out=torch.zeros([N, 3]), reduce='sum') + alphainv_last[..., None] * render_kwargs['bg']
            k_rgb_marched = k_rgb_marched.clamp(0, 1)
        else:
            k_rgb_marched = None
            
        # Ray marching
        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id, out=torch.zeros([N, 3]), reduce='sum') + alphainv_last[..., None] * render_kwargs['bg']

        if gradient is not None and render_kwargs.get('render_grad', False):
            normal = gradient / (gradient.norm(2, -1, keepdim=True) + 1e-6)
            normal_marched = segment_coo(
                src=(weights.unsqueeze(-1) * normal),
                index=ray_id, out=torch.zeros([N, 3]), reduce='sum')
        else:
            normal_marched = None

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * step_id * dist), index=ray_id, out=torch.zeros([N]), reduce='sum')
            disp = 1 / depth
        else:
            depth = None
            disp = 0
        ret_dict.update({
            'alphainv_cum': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            # 'k_rgb_marched': k_rgb_marched,
            'normal_marched': normal_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask,
            'mask_outbbox':mask_outbbox,
            'gradient': gradient,
            "gradient_error": gradient_error,
            "s_val": s_val,
        })
        if self.use_rgb_k:
            ret_dict.update({
            'rgb_marched': k_rgb_marched,
            'rgb_marched0': rgb_marched,
            })
        return ret_dict

    def mesh_color_forward(self, ray_pts, **kwargs):

        sdf_grid = self.smooth_conv(
            self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        # self.gradient = self.neus_sdf_gradient()
        sdf, gradient, feat = self.grid_sampler(ray_pts, sdf_grid, sample_ret=True, sample_grad=True, displace=1.0)
        normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
        viewdirs = -normal

        k0 = self.k0(ray_pts)

        all_grad_inds = list(set(self.grad_feat + self.k_grad_feat))
        all_sdf_inds = list(set(self.sdf_feat + self.k_sdf_feat))

        assert all_grad_inds == all_sdf_inds

        if len(all_grad_inds) > 0:
            all_grad_inds = sorted(all_grad_inds)
            all_grad_inds_ = deepcopy(all_grad_inds)
            all_feat, all_grad = self.sample_sdfs(ray_pts, sdf_grid,
                                                  displace_list=all_grad_inds_,
                                                  use_grad_norm=self.use_grad_norm)
        else:
            all_feat, all_grad = None, None

        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat(
            [viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        if self.use_rgbnet_k0:
            rgb_feat = torch.cat([
                k0, xyz_emb, viewdirs_emb.flatten(0, -2)
            ], -1)
        else:
            rgb_feat = torch.cat([
                xyz_emb, viewdirs_emb.flatten(0, -2)
            ], -1)

        hierarchical_feats = []
        if self.center_sdf:
            hierarchical_feats.append(sdf[:, None])
        if len(all_grad_inds) > 0:
            hierarchical_feats.append(all_feat)
            hierarchical_feats.append(all_grad)
        if len(hierarchical_feats) > 0:
            rgb_feat = torch.cat([rgb_feat, *hierarchical_feats], dim=-1)

        rgb_logit = self.rgbnet(rgb_feat)
        rgb = torch.sigmoid(rgb_logit)

        if self.use_rgb_k:
            k_xyz_emb = (rays_xyz.unsqueeze(-1) * self.k_posfreq).flatten(-2)
            k_xyz_emb = torch.cat([rays_xyz, k_xyz_emb.sin(), k_xyz_emb.cos()],
                                  -1)
            k_viewdirs_emb = (viewdirs.unsqueeze(-1) * self.k_viewfreq).flatten(
                -2)
            k_viewdirs_emb = torch.cat(
                [viewdirs, k_viewdirs_emb.sin(), k_viewdirs_emb.cos()], -1)
            k_rgb_feat = torch.cat([
                k0, k_xyz_emb, k_viewdirs_emb.flatten(0, -2)
            ], -1)

            assert len(self.k_grad_feat) == 1 and self.k_grad_feat[0] == 1.0
            assert len(self.k_sdf_feat) == 0
            all_feats_ = [gradient]

            if self.k_center_sdf:
                all_feats_.append(sdf[:, None])
            if len(all_feats_) > 0:
                all_feats_ = torch.cat(all_feats_, dim=-1)
                k_rgb_feat = torch.cat([k_rgb_feat, all_feats_], dim=-1)

            if self.k_res:
                color_feat = rgb_logit
                if self.k_detach_1:
                    k_rgb_feat = torch.cat([k_rgb_feat, color_feat.detach()], dim=-1)
                else:
                    k_rgb_feat = torch.cat([k_rgb_feat, color_feat], dim=-1)

            if self.k_detach_2:
                k_rgb_logit = rgb_logit.detach() + self.k_rgbnet(k_rgb_feat)
            else:
                k_rgb_logit = rgb_logit + self.k_rgbnet(k_rgb_feat)
            rgb = torch.sigmoid(k_rgb_logit)

        return rgb

    def extract_geometry(self, bound_min, bound_max, resolution=128, threshold=0.0, smooth=True, sigma=0.5, **kwargs):
        if self.smooth_sdf:
            sdf_grid = self.smooth_conv(self.sdf.grid)
        else:
            if smooth:
                self.init_smooth_conv_test_k3(sigma=sigma)
                sdf_grid = self.smooth_conv_test_k3(self.sdf.grid)
            else:
                sdf_grid = self.sdf.grid
        query_func = lambda pts: self.grid_sampler(pts, - sdf_grid)
        if resolution is None:
            resolution = self.world_size[0]
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=query_func)



''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, ks=3):
        super().__init__()
        st = torch.load(path)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        self.register_buffer('density', F.max_pool3d(
            st['model_state_dict']['density'], kernel_size=ks, padding=ks//2, stride=1))
        self.act_shift = st['MaskCache_kwargs']['act_shift']
        self.voxel_size_ratio = st['MaskCache_kwargs']['voxel_size_ratio']
        self.nearest = st['MaskCache_kwargs'].get('nearest', False)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode='nearest')
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return (alpha >= self.mask_cache_thres)


''' Misc
'''
def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[...,[0]]), p.clamp_min(1e-10).cumprod(-1)], -1)

def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1-alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum

def total_variation(v, mask=None):
    if torch.__version__ == '1.10.0':
        tv2 = v.diff(dim=2).abs()
        tv3 = v.diff(dim=3).abs()
        tv4 = v.diff(dim=4).abs()
    else:
        tv2 = (v[:,:,1:,:,:] - v[:,:,:-1,:,:]).abs()
        tv3 = (v[:,:,:,1:,:] - v[:,:,:,:-1,:]).abs()
        tv4 = (v[:,:,:,:,1:] - v[:,:,:,:,:-1]).abs()
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

def total_variation_step2(v, mask=None):
    tv2 = (v[:,:,2:,:,:] - v[:,:,:-2,:,:]).abs() / 2
    tv3 = (v[:,:,:,2:,:] - v[:,:,:,:-2,:]).abs() / 2
    tv4 = (v[:,:,:,:,2:] - v[:,:,:,:,:-2]).abs() / 2
    if mask is not None:
        tv2 = tv2[mask[:,:,:-2] & mask[:,:,2:]]
        tv3 = tv3[mask[:,:,:,:-2] & mask[:,:,:,2:]]
        tv4 = tv4[mask[:,:,:,:,:-2] & mask[:,:,:,:,2:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None

''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs, rgbnet_sup_reduce=1):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz



def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS


