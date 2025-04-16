def extra_repr(self):
    st = ''
    if hasattr(self.self_attn, 'dropout'):
        st += f'attn_dr={self.self_attn.dropout}'
    return st
