weight = None
resume = False
evaluate = True
test_only = False
seed = 10115305
save_path = './exp/debug'
num_worker = 16
batch_size = 2
batch_size_val = None
batch_size_test = None
epoch = 100
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = True
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0001)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentorV2',
    num_classes=7,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo')),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'NMGDataset'
data_root = '/nas2/YJ/DATA/NMG/3D_RAW_7/cropped1024'
ignore_index = -1
names = [
    'Pylon', 'Powerline', 'HighVegetation', 'MediumVegetation', 'Ground',
    'Building', 'Others'
]
data = dict(
    num_classes=7,
    ignore_index=-1,
    names=[
        'Pylon', 'Powerline', 'HighVegetation', 'MediumVegetation', 'Ground',
        'Building', 'Others'
    ],
    train=dict(
        type='NMGDataset',
        split='train',
        data_root='/nas2/YJ/DATA/NMG/3D_RAW_7/cropped1024',
        transform=[
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(
                type='GridSample',
                grid_size=3.0,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', ))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='NMGDataset',
        split='test',
        data_root='/nas2/YJ/DATA/NMG/3D_RAW_7/cropped1024',
        transform=[
            dict(
                type='GridSample',
                grid_size=3.0,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', ))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='NMGDataset',
        split='test',
        data_root='/nas2/YJ/DATA/NMG/3D_RAW_7/cropped1024',
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=3.0,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'segment'),
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=3.0,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys='coord'),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', ))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [1, 1]
            }]]),
        ignore_index=-1))
