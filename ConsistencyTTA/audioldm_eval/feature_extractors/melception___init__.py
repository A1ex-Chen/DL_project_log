def __init__(self, num_classes, features_list,
    feature_extractor_weights_path, **kwargs):
    super().__init__(num_classes=num_classes, init_weights=True, **kwargs)
    self.features_list = list(features_list)
    self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
    self.maxpool1 = torch.nn.Identity()
    self.maxpool2 = torch.nn.Identity()
    state_dict = torch.load(feature_extractor_weights_path, map_location='cpu')
    self.load_state_dict(state_dict['model'])
    for p in self.parameters():
        p.requires_grad_(False)
