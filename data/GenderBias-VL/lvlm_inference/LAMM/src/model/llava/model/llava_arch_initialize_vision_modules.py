def initialize_vision_modules(self, model_args, fsdp=None):
    vision_tower = model_args.vision_tower
    mm_vision_select_layer = model_args.mm_vision_select_layer
    mm_vision_select_feature = model_args.mm_vision_select_feature
    pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
    self.config.mm_vision_tower = vision_tower
    if self.get_vision_tower() is None:
        vision_tower = build_vision_tower(model_args)
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower
    else:
        if fsdp is not None and len(fsdp) > 0:
            vision_tower = self.vision_tower[0]
        else:
            vision_tower = self.vision_tower
        vision_tower.load_model()
    self.config.use_mm_proj = True
    self.config.mm_projector_type = getattr(model_args, 'mm_projector_type',
        'linear')
    self.config.mm_hidden_size = vision_tower.hidden_size
    self.config.mm_vision_select_layer = mm_vision_select_layer
    self.config.mm_vision_select_feature = mm_vision_select_feature
    if getattr(self, 'mm_projector', None) is None:
        self.mm_projector = build_vision_projector(self.config)
    else:
        for p in self.mm_projector.parameters():
            p.requires_grad = True
    if pretrain_mm_mlp_adapter is not None:
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter,
            map_location='cpu')

        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items(
                ) if keyword in k}
        self.mm_projector.load_state_dict(get_w(mm_projector_weights,
            'mm_projector'))
