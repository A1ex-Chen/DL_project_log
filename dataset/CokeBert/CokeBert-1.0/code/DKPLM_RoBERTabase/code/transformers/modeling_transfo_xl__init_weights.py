def _init_weights(self, m):
    """ Initialize the weights.
        """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            self._init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            self._init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, self.config.
                        proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            self._init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            self._init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            self._init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, self.config.
                        proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, self.config.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            self._init_bias(m.bias)
    else:
        if hasattr(m, 'r_emb'):
            self._init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            self._init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            self._init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            self._init_bias(m.r_bias)
