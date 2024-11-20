def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
    pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
    self.config.mm_vision_tower = vision_tower
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    if not hasattr(self, 'vision_tower'):
        vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
    else:
        vision_tower = self.vision_tower
    vision_tower.requires_grad_(False)
    vision_tower = vision_tower.to(torch.float16)
    self.vision_tower = vision_tower
    vision_config = vision_tower.config
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    self.config.use_mm_proj = True
    self.config.mm_hidden_size = vision_config.hidden_size
    self.config.mm_vision_select_layer = mm_vision_select_layer
    if not hasattr(self, 'mm_projector'):
        self.mm_projector = nn.Linear(vision_config.hidden_size, self.
            config.hidden_size)
    if pretrain_mm_mlp_adapter is not None:
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter,
            map_location='cpu')
        self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in
            mm_projector_weights.items()})
    return dict(image_processor=image_processor, image_token_len=
        num_patches, vision_config=vision_config)
