def __init__(self):
    self.transform = None
    import albumentations as A
    self.transform = A.Compose([A.CLAHE(p=0.01), A.RandomBrightnessContrast
        (brightness_limit=0.2, contrast_limit=0.2, p=0.01), A.RandomGamma(
        gamma_limit=[80, 120], p=0.01), A.Blur(p=0.01), A.MedianBlur(p=0.01
        ), A.ToGray(p=0.01), A.ImageCompression(quality_lower=75, p=0.01)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=[
        'class_labels']))
