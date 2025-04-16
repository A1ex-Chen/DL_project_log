def __init__(self):
    self.transform = None
    try:
        import albumentations as A
        check_version(A.__version__, '1.0.3', hard=True)
        T = [A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.
            CLAHE(p=0.01), A.RandomBrightnessContrast(p=0.0), A.RandomGamma
            (p=0.0), A.ImageCompression(quality_lower=75, p=0.0)]
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format=
            'yolo', label_fields=['class_labels']))
        LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in
            self.transform.transforms if x.p))
    except ImportError:
        pass
    except Exception as e:
        LOGGER.info(colorstr('albumentations: ') + f'{e}')
