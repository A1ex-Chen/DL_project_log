def init_text_branch_parameters(self):
    if self.text_branch_type == 'transformer':
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = self.text_branch.width ** -0.5 * (2 * self.text_branch.
            layers) ** -0.5
        attn_std = self.text_branch.width ** -0.5
        fc_std = (2 * self.text_branch.width) ** -0.5
        for block in self.text_branch.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    if self.text_branch_type == 'bert' or self.text_branch_type == 'roberta':
        width = self.text_branch.embeddings.word_embeddings.weight.shape[-1]
    elif self.text_branch_type == 'bart':
        width = self.text_branch.shared.weight.shape[-1]
    else:
        width = self.text_branch.width
    nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))
    nn.init.constant_(self.logit_scale_t, np.log(1 / 0.07))
