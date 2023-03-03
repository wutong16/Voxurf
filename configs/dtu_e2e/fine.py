_base_ = '../default_fine_s.py'

expname = 'scan'
basedir = './logs/dtu'
train_all = False
reso_level = 1
exp_stage = 'fine'

use_sp_color = True
white_list = [24, 40, 110]
black_list = [37, 55, 63, 65, 69, 83, 97, 105, 106, 114, 118, 122]

data = dict(
    datadir='./data/DTU/dtu_scan',
    dataset_type='dtu',
    inverse_y=True,
    white_bkgd=False,
)

surf_train=dict(
    load_density_from=None,
    load_sdf_from='auto',

    pg_scale=[15000],
    scale_ratio=4.096,
    weight_rgb0=0.5,  # this is for the first rgbnet
    weight_main=1,    # this is for k_rgbnet, which is the final output

    sdf_reduce=0.3,
    N_iters=20000,
    lrate_decay=20,
    # eval_iters=[100, 500, 1000, 2000, 3000, 5000, 10000, 15000, 17000, 18000, 19000, 20000, 25000, 30000, 35000],

    tv_dense_before=20000,
    tv_end=30000,
    tv_every=3,
    weight_tv_density=0.01,
    tv_terms=dict(
        sdf_tv=0.1,
        grad_tv=0,
        grad_norm=0,
        smooth_grad_tv=0.05,
    ),
    cosine_lr=True,
    cosine_lr_cfg=dict(
        warm_up_iters=0,
        const_warm_up=True,
        warm_up_min_ratio=1.0),
    lrate_sdf=5e-3,
    decay_step_module={
        15000:dict(sdf=0.1),
    },
    lrate_k0=1e-1, #1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3 * 3, # 1e-3,       # lr of the mlp to preduct view-dependent color
    lrate_k_rgbnet=1e-3,
)

surf_model_and_render=dict(
    num_voxels=256**3,
    num_voxels_base=256**3,
    posbase_pe=5,
    viewbase_pe=1,
    k_posbase_pe=5, # default = 5
    k_viewbase_pe=1, # default = 4
    k_res=True, # default = True
    rgbnet_full_implicit=False, # by using a full mlp without local feature for rgb, the info for the geometry would be better
    rgbnet_depth=4,
    k_rgbnet_depth=4, # deeper is better
    k_grad_feat=(1.0,),  # default = 0 | or set as 3 to feed in the normal itself | or set as geo_rgb_dim to feed in the hierachical normal
    k_sdf_feat=(), # default = 0 | or we could set it as feat_rgb_dim so that it takes in the feature
    rgbnet_dim=6, # larger is better
    rgbnet_width=192,
    center_sdf=True,
    k_center_sdf=False,
    grad_feat=(0.5, 1.0, 1.5, 2.0,),
    sdf_feat=(0.5, 1.0, 1.5, 2.0,),
    octave_use_corner=False,
    use_grad_norm=True,
    use_mlp_residual=False,
    surface_sampling=False,
    use_trimap=False,
    n_importance=64,
    up_sample_steps=1,
    stepsize=0.5,
    s_ratio=50,
    s_start=0.05,
)
