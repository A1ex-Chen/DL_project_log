def initialize_parameters(self):
    nn.init.normal_(self.token_embedding.weight, std=0.02)
    nn.init.normal_(self.positional_embedding, std=0.01)
    if isinstance(self.visual, ModifiedResNet):
        if self.visual.attnpool is not None:
            std = self.visual.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        for resnet_block in [self.visual.layer1, self.visual.layer2, self.
            visual.layer3, self.visual.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith('bn3.weight'):
                    nn.init.zeros_(param)
    proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers
        ) ** -0.5
    attn_std = self.transformer.width ** -0.5
    fc_std = (2 * self.transformer.width) ** -0.5
    for block in self.transformer.resblocks:
        nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    if self.text_projection is not None:
        nn.init.normal_(self.text_projection, std=self.transformer.width **
            -0.5)
