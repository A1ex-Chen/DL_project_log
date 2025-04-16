def _build_net(self) ->nn.Module:
    resnest269 = resnest.torch.resnest269(pretrained=self.pretrained)
    resnest269.fc = nn.Linear(in_features=resnest269.fc.in_features,
        out_features=self.num_classes)
    children = list(resnest269.children())
    conv1 = nn.Sequential(*children[:3])
    conv2 = nn.Sequential(*children[3:5])
    conv3 = children[5]
    conv4 = children[6]
    conv5 = children[7]
    modules = [conv1, conv2, conv3, conv4, conv5]
    assert 0 <= self.num_frozen_levels <= len(modules)
    freezing_modules = modules[:self.num_frozen_levels]
    for module in freezing_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return resnest269
