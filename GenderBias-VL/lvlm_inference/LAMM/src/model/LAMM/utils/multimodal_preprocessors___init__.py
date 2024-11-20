def __init__(self, kernel_size: int, imu_stem: PatchEmbedGeneric, embed_dim:
    int, img_size: List=(6, 2000), num_cls_tokens: int=1, pos_embed_fn:
    Callable=None, init_param_style: str='openclip') ->None:
    super().__init__()
    stem = imu_stem
    self.imu_stem = imu_stem
    self.embed_dim = embed_dim
    self.use_pos_embed = pos_embed_fn is not None
    self.num_cls_tokens = num_cls_tokens
    self.kernel_size = kernel_size
    self.pos_embed = nn.Parameter(torch.empty(1, img_size[1] // kernel_size +
        num_cls_tokens, embed_dim))
    if self.num_cls_tokens > 0:
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_cls_tokens,
            self.embed_dim))
    self.init_parameters(init_param_style)
