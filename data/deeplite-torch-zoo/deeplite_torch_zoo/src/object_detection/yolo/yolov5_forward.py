def forward(self, x, augment=False, profile=False, visualize=False):
    if augment:
        return self._forward_augment(x)
    return self._forward_once(x, profile, visualize)
