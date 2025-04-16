def prepare_resnet(self):
    ret = deepcopy(self)
    ret.stages = nn.ModuleList(ret.stages)
    for k in self.stage_names:
        delattr(ret, k)
    return ret
