def classify_albumentations(augment=True, size=224, scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75), hflip=0.5, vflip=0.0, jitter=0.4, mean=
    IMAGENET_MEAN, std=IMAGENET_STD, auto_aug=False):
    prefix = colorstr('albumentations: ')
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        check_version(A.__version__, '1.0.3', hard=True)
        if augment:
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale,
                ratio=ratio)]
            if auto_aug:
                LOGGER.info(
                    f'{prefix}auto augmentations are currently not supported')
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size,
                width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]
        LOGGER.info(prefix + ', '.join(f'{x}'.replace(
            'always_apply=False, ', '') for x in T if x.p))
        return A.Compose(T)
    except ImportError:
        LOGGER.warning(
            f'{prefix}⚠️ not found, install with `pip install albumentations` (recommended)'
            )
    except Exception as e:
        LOGGER.info(f'{prefix}{e}')
