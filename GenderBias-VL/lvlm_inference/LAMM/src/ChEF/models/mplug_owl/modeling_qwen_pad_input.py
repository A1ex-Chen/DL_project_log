def pad_input(self, hidden_states, indices, batch, seqlen):
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=
        hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, '(b s) ... -> b s ...', b=batch)
