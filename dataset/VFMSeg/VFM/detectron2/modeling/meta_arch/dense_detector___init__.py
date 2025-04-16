def __init__(self, backbone: Backbone, head: nn.Module, head_in_features:
    Optional[List[str]]=None, *, pixel_mean, pixel_std):
    """
        Args:
            backbone: backbone module
            head: head module
            head_in_features: backbone features to use in head. Default to all backbone features.
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        """
    super().__init__()
    self.backbone = backbone
    self.head = head
    if head_in_features is None:
        shapes = self.backbone.output_shape()
        self.head_in_features = sorted(shapes.keys(), key=lambda x: shapes[
            x].stride)
    else:
        self.head_in_features = head_in_features
    self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1,
        1), False)
    self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1
        ), False)
