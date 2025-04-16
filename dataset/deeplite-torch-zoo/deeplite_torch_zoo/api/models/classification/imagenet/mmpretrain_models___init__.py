def __init__(self, model_name, pretrained=False, num_classes=
    NUM_IMAGENET_CLASSES, dummy_input_size=(2, 3, 224, 224)):
    super().__init__()
    self.model = mmpretrain.get_model(model_name, pretrained=pretrained,
        device='cpu')
    device = next(self.model.parameters()).device
    if hasattr(self.model.backbone, 'img_size'):
        dummy_input_size = 2, 3, *self.model.backbone.img_size
    if num_classes != NUM_IMAGENET_CLASSES:
        head_type = self.model.head.__class__
        if hasattr(self.model.head, 'fc'):
            feature_dim = self.model.head.fc.in_features
        else:
            feature_dim = self.model.extract_feat(torch.randn(
                dummy_input_size).to(device)).shape[1]
        LOGGER.warning(
            f'Replacing MMPretrain model head with a {head_type} head with in_channels={{feature_dim}} and num_classes={{num_classes}}'
            )
        self.model.head = head_type(in_channels=feature_dim, num_classes=
            num_classes).to(device)
