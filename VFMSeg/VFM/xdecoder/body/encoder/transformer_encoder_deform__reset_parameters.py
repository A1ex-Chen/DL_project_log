def _reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for m in self.modules():
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    normal_(self.level_embed)
