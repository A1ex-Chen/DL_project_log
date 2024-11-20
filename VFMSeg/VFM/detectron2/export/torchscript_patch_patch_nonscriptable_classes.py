def patch_nonscriptable_classes():
    """
    Apply patches on a few nonscriptable detectron2 classes.
    Should not have side-effects on eager usage.
    """
    from detectron2.modeling.backbone import ResNet, FPN

    def prepare_resnet(self):
        ret = deepcopy(self)
        ret.stages = nn.ModuleList(ret.stages)
        for k in self.stage_names:
            delattr(ret, k)
        return ret
    ResNet.__prepare_scriptable__ = prepare_resnet

    def prepare_fpn(self):
        ret = deepcopy(self)
        ret.lateral_convs = nn.ModuleList(ret.lateral_convs)
        ret.output_convs = nn.ModuleList(ret.output_convs)
        for name, _ in self.named_children():
            if name.startswith('fpn_'):
                delattr(ret, name)
        return ret
    FPN.__prepare_scriptable__ = prepare_fpn
    from detectron2.modeling.roi_heads import StandardROIHeads
    if hasattr(StandardROIHeads, '__annotations__'):
        StandardROIHeads.__annotations__ = deepcopy(StandardROIHeads.
            __annotations__)
        StandardROIHeads.__annotations__['mask_on'] = torch.jit.Final[bool]
        StandardROIHeads.__annotations__['keypoint_on'] = torch.jit.Final[bool]
