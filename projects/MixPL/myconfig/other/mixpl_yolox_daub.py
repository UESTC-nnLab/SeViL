_base_ = [
    'mmdet::_base_/default_runtime.py', '../mixpl_daub_detection.py'
]

custom_imports = dict(
    imports=['projects.MixPL.mixpl'], allow_failed_imports=False)

detector = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(512, 512),
                size_divisor=32,
                interval=10)
        ]),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33, 
        widen_factor=0.5, 
        out_indices=(2, 3, 4), #(2, 3, 4)
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpt/darknet53-a628ea1b.pth')
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),

        bbox_head=dict(
            type = 'YOLOXHead',
            num_classes=1,
            in_channels=128,
            stacked_convs=2,
            feat_channels=128,
            strides=[4, ],#[4, 8, 16]
            use_depthwise=False,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish'),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
           loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
            ),
     train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    #  train_cfg=dict(
    #     assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65), #type='soft_nms'
        max_per_img=100))

model = dict( 
    type='MixPL',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        least_num=1,
        cache_size=8,
        mixup=False,
        mosaic=False,
        mosaic_shape=[(400, 400), (800, 800)],
        mosaic_weight=0.5,
        erase=False,
        erase_patches=(1, 10),
        erase_ratio=(0, 0.02),
        erase_thr=0.7,
        cls_pseudo_thr=0.3,
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='student'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
# labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
labeled_dataset.ann_file = 'annotations/instances_train2017.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10-unlabeled.json'
labeled_dataset.data_prefix = dict(img='')
unlabeled_dataset.data_prefix = dict(img='')

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(batch_size=4, source_ratio=[1, 3]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=50000, val_interval=1000,val_begin = 1000) #,val_begin = 3000
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')
 
# learning rate policy 
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2)
)

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=3,save_best='student/coco/bbox_mAP_50',  # 按照该指标保存最优模型
        type='CheckpointHook'))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume=False
seed = 3407