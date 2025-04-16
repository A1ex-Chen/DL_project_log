def lock(self, unlocked_groups=0, freeze_bn_stats=False):
    assert unlocked_groups == 0, 'partial locking not currently supported for this model'
    for param in self.parameters():
        param.requires_grad = False
