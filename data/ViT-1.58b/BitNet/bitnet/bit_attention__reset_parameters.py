def _reset_parameters(self):
    nn.init.xavier_normal_(self.q_proj.weight)
    if self.q_proj.bias is not None:
        nn.init.constant_(self.q_proj.bias, 0)
    nn.init.xavier_normal_(self.k_proj.weight)
    if self.k_proj.bias is not None:
        nn.init.constant_(self.k_proj.bias, 0)
    nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
    if self.v_proj.bias is not None:
        nn.init.constant_(self.v_proj.bias, 0)
    nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
    if self.out_proj.bias is not None:
        nn.init.constant_(self.out_proj.bias, 0)
