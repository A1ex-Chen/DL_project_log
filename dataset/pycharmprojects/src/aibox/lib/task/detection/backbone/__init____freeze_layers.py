def _freeze_layers(self):
    modules = [self.component.conv1, self.component.conv2, self.component.
        conv3, self.component.conv4, self.component.conv5]
    assert 0 <= self.num_frozen_levels <= len(modules)
    freezing_modules = modules[:self.num_frozen_levels]
    for module in freezing_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
