def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order
    =False, **kwargs):
    super(TFAdaptiveSoftmaxMask, self).__init__(**kwargs)
    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.cutoffs = cutoffs + [n_token]
    self.cutoff_ends = [0] + self.cutoffs
    self.div_val = div_val
    self.shortlist_size = self.cutoffs[0]
    self.n_clusters = len(self.cutoffs) - 1
    self.head_size = self.shortlist_size + self.n_clusters
    self.keep_order = keep_order
    self.out_layers = []
    self.out_projs = []
