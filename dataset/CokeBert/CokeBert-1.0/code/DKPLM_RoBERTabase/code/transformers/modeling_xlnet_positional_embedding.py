@staticmethod
def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)],
        dim=-1)
    pos_emb = pos_emb[:, None, :]
    if bsz is not None:
        pos_emb = pos_emb.expand(-1, bsz, -1)
    return pos_emb
