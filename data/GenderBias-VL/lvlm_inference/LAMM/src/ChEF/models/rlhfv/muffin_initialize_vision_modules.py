def initialize_vision_modules(self, vision_tower, no_randaug, num_query):
    self.config.mm_vision_tower = vision_tower
    self.config.use_mm_proj = True
    self.config.num_query = num_query
    if not hasattr(self, 'vision_tower'):
        vision_tower = timm.create_model(vision_tower)
        state_dict = torch.load(
            '/mnt/data/user/tc_agi/multi_modal/checkpoints/beit-v3/beit3_large_patch16_224.pth'
            , map_location='cuda')['model']
        state_dict = load_model_and_may_interpolate(state_dict, vision_tower)
        vision_tower.load_state_dict(state_dict, strict=False)
    else:
        vision_tower = self.vision_tower
    self.vision_tower = vision_tower.to(torch.float16)
    train_img_transform, eval_img_transform = build_transform(is_train=True,
        randaug=not no_randaug, input_size=vision_tower.args.img_size
        ), build_transform(is_train=False, input_size=vision_tower.args.
        img_size)
    if num_query is not None and not hasattr(self, 'query'):
        assert num_query > 2
        bos_weight = vision_tower.beit3.text_embed.weight.data[0]
        eos_weight = vision_tower.beit3.text_embed.weight.data[2]
        query_init_weight = [bos_weight] + [None] * (num_query - 2) + [
            eos_weight]
        self.query = construct_query_parameter(num_query, vision_tower.args
            .encoder_embed_dim, query_init_weight)
    if not hasattr(self, 'mm_projector'):
        self.mm_projector = nn.Linear(vision_tower.hidden_size, self.config
            .hidden_size)
    return dict(image_processor=(train_img_transform, eval_img_transform),
        image_token_len=num_query, vision_config=self.vision_config)
