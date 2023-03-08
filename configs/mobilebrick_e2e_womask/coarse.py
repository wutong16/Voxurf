_base_ = '../default_fine_s.py'

expname = ''
basedir = './logs/mobile_brick'
train_all = True
reso_level = 2
exp_stage = 'coarse'

use_sp_color = False

data = dict(
    datadir='./data/mobile_brick/test/',
    dataset_type='mobile_brick',
    inverse_y=True,
    white_bkgd= False,
    mode=dict(
        train_all=False,
        wmask=False,
    ),
)

surf_train=dict(
    load_density_from='',

    box_size=1.,
    N_iters=10000,
    lrate_decay=20,
    weight_tv_k0=0.01,
    weight_entropy_last=0.0,
    weight_tv_density=0.01,

    ori_tv=True,
    tv_terms=dict(
        sdf_tv=0.1,
        grad_tv=0,
        smooth_grad_tv=0.05,
    ),
    tv_dense_before=20000,

    cosine_lr=True,
    cosine_lr_cfg=dict(
        warm_up_iters=0,
        const_warm_up=True,
        warm_up_min_ratio=1.0),

    lrate_sdf=0.1,

    lrate_k0=1e-1, #1e-1,                # lr of color/feature voxel grid
    lrate_bg_k0=1e-1, #1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3, # 1e-3,           # lr of the mlp to preduct view-dependent color
    lrate_bg_rgbnet=1e-3,
    lrate_bg_density=0.1,
    lrate_density=0,
    ray_sampler='random',

    weight_nearclip=0.,
    weight_distortion=0.,

)

surf_model_and_render=dict(
    num_voxels=96**3,
    num_voxels_base=96**3,
    num_voxels_bg=96**3,
    rgbnet_full_implicit=False, # by using a full mlp without local feature for rgb, the info for the geometry would be better
    rgbnet_depth=3,
    geo_rgb_dim=3,
    combine=False,
    use_bound_mask=False,
    bg_fast_color_thres=1e-4,
    fast_color_thres=1e-4,

    smooth_ksize=5,
    smooth_sigma=0.8,
    sdf_thr=0.5,
    tv_in_sphere=False,
    use_cosine_sdf=True,
    cosine_sdf_mini_ratio=0.1,

    smooth_scale=True,
    s_ratio=200,
    s_start=0.5,
    bg_rgbnet_dim=0,
)
