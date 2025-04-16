def __call__(self, pretrained=False, pretrained_from_file=None, **kwargs):
    assert not (pretrained and pretrained_from_file is not None)
    params = replace(self.model.params, **kwargs)
    model = self.model.constructor(arch=self.model.arch, **asdict(params))
    state_dict = None
    if pretrained:
        assert self.model.checkpoint_url is not None
        state_dict = torch.hub.load_state_dict_from_url(self.model.
            checkpoint_url, map_location=torch.device('cpu'))
    if pretrained_from_file is not None:
        if os.path.isfile(pretrained_from_file):
            print("=> loading pretrained weights from '{}'".format(
                pretrained_from_file))
            state_dict = torch.load(pretrained_from_file, map_location=
                torch.device('cpu'))
        else:
            print("=> no pretrained weights found at '{}'".format(
                pretrained_from_file))
    if state_dict is not None:
        state_dict = {(k[len('module.'):] if k.startswith('module.') else k
            ): v for k, v in state_dict.items()}

        def reshape(t, conv):
            if conv:
                if len(t.shape) == 4:
                    return t
                else:
                    return t.view(t.shape[0], -1, 1, 1)
            elif len(t.shape) == 4:
                return t.view(t.shape[0], t.shape[1])
            else:
                return t
        state_dict = {k: (reshape(v, conv=dict(model.named_modules())['.'.
            join(k.split('.')[:-2])].use_conv) if is_se_weight(k, v) else v
            ) for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model
