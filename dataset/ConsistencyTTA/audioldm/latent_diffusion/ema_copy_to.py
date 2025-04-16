def copy_to(self, model):
    m_param = dict(model.named_parameters())
    shadow_params = dict(self.named_buffers())
    for key in m_param:
        if m_param[key].requires_grad:
            m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data
                )
        else:
            assert not key in self.m_name2s_name
