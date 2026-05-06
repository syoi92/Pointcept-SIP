
_base_ = ["../_base_/default_runtime.py"]
enable_wandb = False
wandb_project = "SIP-Pointcept"  
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="SIPInformationWriter"),
    # dict(type="SemSegEvaluator"),
    dict(type="SIPSemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

# Trainer
# train = dict(type="SIPSceneTrainer")
train = dict(type="SIPFragmentTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)


# misc custom setting
batch_size =  1 # bs: total bs in all gpus
fragment_batch_size = 4
num_worker = 12
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "SIPDataset"
data_root = "data/sip-base-r01-cluttered"
sample_res = 0.09
sampling_mode = "grid"
point_max = 30000
max_radius = 0.0 # 8.0
rare_boost = (5, 6)


# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg50",
        in_channels=6,
        num_classes=7,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=0.5, ignore_index=-1, label_smoothing=0.05),        
        dict(type="FocalLoss", loss_weight=1.0, ignore_index=-1, gamma=2.0, alpha=[0.20, 0.20, 0.20, 0.60, 0.20, 0.60, 0.60]),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5, ignore_index=-1),
    ],
)


# scheduler settings
epoch = 25
max_update= 3500    #6500
optimizer = dict(type="AdamW", lr=2e-2, weight_decay=0.01)
scheduler = dict(type="OneCycleLR", max_lr=2e-2, total_steps=max_update, div_factor=10.0, final_div_factor=100.0,)
eval_epoch = 5



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
            dict(type="RandomDropout", dropout_ratio=0.1, dropout_application_ratio=0.5),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.002, clip=0.01),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        fragmentation=       
            dict(
                type="SceneFragmentation",
                split_mode="train",
                point_max=point_max,
                rare_class_ids=rare_boost, 
                max_radius = max_radius,
            ),    
        post_transform=[
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord", "normal"],
                # feat_keys=["coord", "color", "intensity", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        test_mode=False,
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
                split_mode="test",
                point_max=point_max,
                max_radius = max_radius,
            ),    
        post_transform=[
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord", "normal"],
            ),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        test_mode=True,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="SceneSampling",
                mode=sampling_mode,
                sample_res=sample_res,
                return_grid_coord=True,
                return_inverse=True,
            ),          
        ], 
        fragmentation=       
            dict(
                type="SceneFragmentation",
                split_mode="test",
                point_max=point_max,
                max_radius = max_radius,
            ),    
        post_transform=[     
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "index"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_cfg=dict(
            aug_transform=[[dict(type="Identity")],],
        ),
    ),
)
