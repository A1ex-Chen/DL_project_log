def compute_similarity(self, v_emb, name='default', fake=False):
    if fake:
        return None
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-07)
    t_emb = getattr(self, '{}_text_embeddings'.format(name))
    output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2
        )
    return output
