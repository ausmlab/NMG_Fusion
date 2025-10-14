_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1 #12  # bs: total bs in all gpus
empty_cache = False
enable_amp = True # If True, Nan Error happens when fusing.

# Task Weights
weight_3d = 0.0
weight_2d = 1.0   ## for 2D

# 2D Branch
tile_size = 1024
grid_size = 1.2
num_classes_2d = 2
hidden_dim = 128
return_interm_indices = [2,3,4]
num_select = 300
iou_threshold = 0.5
score_threshold = 0.001

clip_grad = 0.1
# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=64,
    weight_3d = weight_3d,
    weight_2d = weight_2d, ## for 2D
    num_classes2d = 2, ## for 2D
    aux_loss = True, ## for 2D
    set_cost_class = 2.0, ## for 2D
    set_cost_bbox = 5.0, ## for 2D
    set_cost_giou = 2.0, ## for 2D
    cls_loss_coef = 5.0, ## for 2D
    bbox_loss_coef = 5.0, ## for 2D
    giou_loss_coef = 2.0, ## for 2D
    interm_loss_coef = 1.0, ## for 2D
    no_interm_box_loss = False, ## for 2D
    focal_alpha = 0.25, ## for 2D
    dec_layers = 3, ## for 2D
    use_dn = True, ## for 2D
    two_stage_type = 'standard', ## for 2D
    num_select = 300, ## for 2D
    nms_iou_threshold = -1, ## for 2D
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
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
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
        # For 2D Branch
        weight_2d = weight_2d,
        num_classes_2d = num_classes_2d,
        return_interm_indices = return_interm_indices,
        dn_number = 100,
        dn_labelbook_size = 2,
        dn_label_noise_ratio = 0.5,
        dn_box_noise_scale = 1.0,
        pe_temperatureH = 20,
        pe_temperatureW = 20,
        aux_loss=True,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=False,
        two_stage_bbox_embed_share=False,
        d_model=hidden_dim,
        dropout=0.0,
        nhead=8,
        num_queries=600,
        dim_feedforward=2048,   ####<-------
        num_encoder_layers=3,
        num_unicoder_layers=0,
        num_decoder_layers=3,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=5,
        activation='relu',
        num_patterns=0,
        enc_n_points=4,
        dec_n_points=4,
        use_deformable_box_attn=False,
        box_attn_type='roi_align',
        decoder_query_perturber=None,
        add_channel_attention=False,
        add_pos_value=False,
        random_refpoints_xy=False,

        # two stage
        tile_size=tile_size,
        grid_size=grid_size,
        two_stage_type='standard',  # ['no', 'standard', 'early']
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        dec_layer_number=None,
        decoder_sa_type='sa',
        module_seq=['sa', 'ca', 'ffn'],

        embed_init_tgt=True,
        use_detached_boxes_dec_out=False
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 500
eval_epoch = 500
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05, eps=1e-7)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001, 0.0001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001),
               dict(keyword="input_proj", lr=0.0001, weight_decay=0.0001),
               dict(keyword="transformer", lr=0.0001, weight_decay=0.0001)]

# dataset settings
dataset_type = "NMGDataset"
data_root = "/nas2/YJ/DATA/NMG/3D_RAW_7/cropped1024"
ignore_index = -1
names = [
    "Pylon",
    "Powerline",
    "HighVegetation",
    "MediumVegetation",
    "Ground",
    "Building",
    "Others",
]

data = dict(
    num_classes=7,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            dict(type="NormalizeOBB"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "target"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            dict(type="NormalizeOBB"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "target"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord"),
            ),
            crop=None,
            post_transform=[
                dict(type="NormalizeOBB"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "target"),
                    feat_keys=("coord",),
                ),
            ],
            #aug_transform=[
            #    [dict(type="RandomScale", scale=[1, 1])],
            #],
        ),
        ignore_index=ignore_index,
    ),
)
