def prepare_fpn(self):
    ret = deepcopy(self)
    ret.lateral_convs = nn.ModuleList(ret.lateral_convs)
    ret.output_convs = nn.ModuleList(ret.output_convs)
    for name, _ in self.named_children():
        if name.startswith('fpn_'):
            delattr(ret, name)
    return ret
