def _build_net(self) ->nn.Module:
    mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=self.pretrained)
    mobilenet_v2.classifier = nn.Sequential(mobilenet_v2.classifier[0], nn.
        Linear(in_features=mobilenet_v2.classifier[1].in_features,
        out_features=self.num_classes))
    children = mobilenet_v2.features
    conv1 = children[:2]
    conv2 = children[2:4]
    conv3 = children[4:7]
    conv4 = children[7:14]
    conv5 = children[14:]
    modules = [conv1, conv2, conv3, conv4, conv5]
    assert 0 <= self.num_frozen_levels <= len(modules)
    freezing_modules = modules[:self.num_frozen_levels]
    for module in freezing_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return mobilenet_v2
