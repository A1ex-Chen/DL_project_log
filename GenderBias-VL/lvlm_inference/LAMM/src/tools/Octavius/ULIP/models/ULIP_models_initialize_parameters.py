def initialize_parameters(self):
    nn.init.normal_(self.token_embedding.weight, std=0.02)
    nn.init.normal_(self.positional_embedding, std=0.01)
    proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers
        ) ** -0.5
    attn_std = self.transformer.width ** -0.5
    fc_std = (2 * self.transformer.width) ** -0.5
    for block in self.transformer.resblocks:
        nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
    nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
