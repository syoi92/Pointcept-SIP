
_base_ = ["../_base_/default_runtime.py"]
enable_wandb = False
wandb_project = "SIP-Pointcept"  
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="SemSegEvaluator"),
    dict(type="SIPSemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="SIPFragmentTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)


# misc custom setting
batch_size =  1 # bs: total bs in all gpus
fragment_batch_size = 1
num_worker = 12
empty_cache = True
enable_amp = True

# dataset settings
dataset_type = "SIPDataset"
data_root = "data/sip-base-r01-cluttered"

sample_res = 0.06
sampling_mode = "manifold"
frag_mode = "scanbin"
point_max = 30000
rare_boost = (5, 6)

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m1",
        in_channels=6,
        num_classes=7,
        patch_embed_depth=2,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8, 
        enc_depths=(2,6,2),#(2, 4, 2), 
        enc_channels=(96, 192, 384), 
        enc_groups=(12, 24, 48), 
        enc_neighbours=(16, 16, 12), 
        dec_depths=(1, 1, 1),
        dec_channels=(48, 96, 192),
        dec_groups=(6, 12, 24),
        dec_neighbours=(16, 16, 12),
        grid_sizes=(0.09, 0.15, 0.24),
        attn_qkv_bias=True,
        pe_multiplier=True,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        enable_checkpoint=False,
        unpool_backend="interp",  # map / interp
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        # dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        dict(type="FocalLoss", loss_weight=0.5, ignore_index=-1, gamma=1.0),
    ],
)

# scheduler settings
epoch = 80
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
eval_epoch = 10



data = dict(
    num_classes=7,
    ignore_index=-1,
    names=[
        "wall",
        "ceiling",
        "floor",
        "pipes",
        "column",
        "ladder",
        "stair",
    ],
    train=dict(
        type=dataset_type,
        split=("train"),
        data_root=data_root,
        transform=[
            dict(
                type="SceneSampling",
                mode=sampling_mode,
                sample_res=sample_res,
                return_grid_coord=True,
            ),  
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),    
        ],
        fragmentation=       
            dict(
                type="SceneFragmentation",
                mode=frag_mode,
                split_mode="train",
                point_max=point_max,
                rare_class_ids=rare_boost, 
            ),    
        post_transform=[
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord","segment"),
                feat_keys=["coord", "color"],
                # feat_keys=["coord", "color", "intensity", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(
                type="SceneSampling",
                mode=sampling_mode,
                sample_res=sample_res,
                return_grid_coord=True,
            ),          
        ],
        fragmentation=       
            dict(
                type="SceneFragmentation",
                mode=frag_mode,
                split_mode="test",
                point_max=point_max,
            ),    
        post_transform=[
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord", "color"],
                # feat_keys=["coord", "color", "intensity", "normal"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        test_mode=True,
        transform=[
            dict(
                type="SceneSampling",
                mode=sampling_mode,
                sample_res=sample_res,
                return_grid_coord=True,
            ),          
        ], 
        fragmentation=       
            dict(
                type="SceneFragmentation",
                mode=frag_mode,
                split_mode="test",
                point_max=point_max,
            ),    
        post_transform=[     
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "index"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_cfg=dict(
            aug_transform=[[dict(type="Identity")],],
            # aug_transform=[
            #     [dict(type="RandomScale", scale=[0.95, 0.95])],
            #     [dict(type="RandomScale", scale=[1, 1])],
            #     [dict(type="RandomScale", scale=[1.05, 1.05])],
            #     [
            #         dict(type="RandomScale", scale=[0.95, 0.95]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            #     [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
            #     [
            #         dict(type="RandomScale", scale=[1.05, 1.05]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            # ],
        ),
    ),
)
