def forward(self, model):
    decay = self.decay
    if self.num_updates >= 0:
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.
            num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                sname = self.m_name2s_name[key]
                shadow_params[sname] = shadow_params[sname].type_as(m_param
                    [key])
                shadow_params[sname].sub_(one_minus_decay * (shadow_params[
                    sname] - m_param[key]))
            else:
                assert not key in self.m_name2s_name
