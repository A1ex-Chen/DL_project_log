# YOLOv6s model
model = dict(
    type='YOLOv6s',
    pretrained='./assets/yolov6s_v2_reopt_43.1.pt',
    scales='./assets/yolov6s_v2_scale.pt',
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        ),
    neck=dict(
        type='RepPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type = 'giou',
        use_dfl = False,
        reg_max = 0,  # if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.00001,
    lrf=0.001,
    momentum=0.937,
    weight_decay=0.00005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

ptq = dict(
    num_bits = 8,
    calib_batches = 4,
    # 'max', 'histogram'
    calib_method = 'histogram',
    # 'entropy', 'percentile', 'mse'
    histogram_amax_method='entropy',
    histogram_amax_percentile=99.99,
    calib_output_path='./',
    sensitive_layers_skip=False,
    sensitive_layers_list=['detect.stems.0.conv',
                           'detect.stems.1.conv',
                           'detect.stems.2.conv',
                           'detect.cls_convs.0.conv',
                           'detect.cls_convs.1.conv',
                           'detect.cls_convs.2.conv',
                           'detect.reg_convs.0.conv',
                           'detect.reg_convs.1.conv',
                           'detect.reg_convs.2.conv',
                           'detect.cls_preds.0',
                           'detect.cls_preds.1',
                           'detect.cls_preds.2',
                           'detect.reg_preds.0',
                           'detect.reg_preds.1',
                           'detect.reg_preds.2',
                           ],
)

qat = dict(
    calib_pt = './assets/yolov6s_v2_reopt_43.1_calib_histogram.pt',
    sensitive_layers_skip = False,
    sensitive_layers_list=['detect.stems.0.conv',
                           'detect.stems.1.conv',
                           'detect.stems.2.conv',
                           'detect.cls_convs.0.conv',
                           'detect.cls_convs.1.conv',
                           'detect.cls_convs.2.conv',
                           'detect.reg_convs.0.conv',
                           'detect.reg_convs.1.conv',
                           'detect.reg_convs.2.conv',
                           'detect.cls_preds.0',
                           'detect.cls_preds.1',
                           'detect.cls_preds.2',
                           'detect.reg_preds.0',
                           'detect.reg_preds.1',
                           'detect.reg_preds.2',
                           ],
)

# Choose Rep-block by the training Mode, choices=["repvgg", "hyper-search", "repopt"]
training_mode='repopt'