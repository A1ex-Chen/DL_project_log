@torch.no_grad()
def init_parameters(self, init_param_style):
    nn.init.normal_(self.pos_embed, std=0.01)
    if init_param_style == 'openclip':
        scale = self.embed_dim ** -0.5
        if self.num_cls_tokens > 0:
            nn.init.normal_(self.cls_token)
            self.cls_token *= scale
    elif init_param_style == 'vit':
        self.cls_token.data.fill_(0)
    else:
        raise ValueError(f'Unknown init {init_param_style}')
