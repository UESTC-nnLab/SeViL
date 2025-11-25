_base_ = [
    'mmdet::_base_/default_runtime.py',
    'mixpl_daub_detection.py'
]

custom_imports = dict(
    imports=['projects.MixPL.mixpl'], allow_failed_imports=False)


detector = dict(
    type="RetinaNet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type="RetinaHead",
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=8,
            scales_per_octave=1,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            # activated=True,# use probability instead of logit as input ConsistentTeacher
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type="PseudoSampler"),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100))


model = dict(
    _delete_=True,
    type='MixPL',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector['data_preprocessor']),
    semi_train_cfg=dict(
        compile=True,
        least_num=1,
        cache_size=8,
        mixup=False,
        mosaic=False,
        mosaic_shape=[(400, 400), (800, 800)],
        mosaic_weight=0.5,
        erase=False,
        erase_patches=(1, 10),
        erase_ratio=(0, 0.01),
        erase_thr=0.7,
        cls_pseudo_thr=0.4,
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=2.0,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10-unlabeled.json'
labeled_dataset.data_prefix = dict(img='train2017/')
unlabeled_dataset.data_prefix = dict(img='train2017/')

train_dataloader = dict(
    batch_size=5,
    num_workers=5,
    sampler=dict(batch_size=5, source_ratio=[1, 4]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
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
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=1))
log_processor = dict(by_epoch=False)
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0002, gamma=4)]
resume=True
