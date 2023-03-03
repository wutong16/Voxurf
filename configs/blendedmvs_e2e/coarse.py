_base_ = '../default_fine_s.py'

expname = 'scan'
basedir = './logs/blended_mvs'
train_all = True
reso_level = 2
exp_stage = 'coarse'


data = dict(
    datadir='./data/BlendedMVS/',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=True, # need to manually adjust the background color for BlendedMVS
)

surf_train=dict(
    load_density_from='',

    pg_filter=[1000,],
    
    tv_add_grad_new=True,
    ori_tv=True,
    weight_main=1,    # this is for rgb_add
    N_iters=10000,
    lrate_decay=20,
    weight_tv_k0=0.01,
    weight_tv_density=0.001,
    tv_terms=dict(
        sdf_tv=0.1,
        grad_tv=0,
        smooth_grad_tv=0.05,
    ),
    tv_updates={
        1000:dict(
            sdf_tv=0.1,
            # grad_tv=10,
            smooth_grad_tv=0.2
        ),
    },
    tv_dense_before=20000,

    lrate_sdf=0.1,
    decay_step_module={
        1000:dict(sdf=0.1),
        5000:dict(sdf=0.5),
    },

    lrate_k0=1e-1, #1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3, # 1e-3,           # lr of the mlp to preduct view-dependent color
    lrate_rgb_addnet=1e-3, # 1e-3,
)

surf_model_and_render=dict(
    num_voxels=96**3,
    num_voxels_base=96**3,
    rgbnet_full_implicit=False, # by using a full mlp without local feature for rgb, the info for the geometry would be better
    posbase_pe=5,
    viewbase_pe=1,
    add_posbase_pe=5,
    add_viewbase_pe=4,
    rgb_add_res=True,
    rgbnet_depth=3,
    geo_rgb_dim=3,

    smooth_ksize=5,
    smooth_sigma=0.8,
    s_ratio=50,
    s_start=0.2,
)
