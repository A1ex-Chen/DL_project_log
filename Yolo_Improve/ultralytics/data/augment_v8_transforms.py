def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste), RandomPerspective(degrees=hyp.degrees,
        translate=hyp.translate, scale=hyp.scale, shear=hyp.shear,
        perspective=hyp.perspective, pre_transform=None if stretch else
        LetterBox(new_shape=(imgsz, imgsz)))])
    flip_idx = dataset.data.get('flip_idx', [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning(
                "WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'"
                )
        elif flip_idx and len(flip_idx) != kpt_shape[0]:
            raise ValueError(
                f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}'
                )
    return Compose([pre_transform, MixUp(dataset, pre_transform=
        pre_transform, p=hyp.mixup), Albumentations(p=1.0), RandomHSV(hgain
        =hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v), RandomFlip(direction
        ='vertical', p=hyp.flipud), RandomFlip(direction='horizontal', p=
        hyp.fliplr, flip_idx=flip_idx)])
