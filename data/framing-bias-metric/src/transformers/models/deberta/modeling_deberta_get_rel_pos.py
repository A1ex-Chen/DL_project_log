def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
    if self.relative_attention and relative_pos is None:
        q = query_states.size(-2
            ) if query_states is not None else hidden_states.size(-2)
        relative_pos = build_relative_position(q, hidden_states.size(-2),
            hidden_states.device)
    return relative_pos
