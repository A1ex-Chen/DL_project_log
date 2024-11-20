def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1,
    keep_order=False, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.cutoffs = cutoffs + [vocab_size]
    self.cutoff_ends = [0] + self.cutoffs
    self.div_val = div_val
    self.shortlist_size = self.cutoffs[0]
    self.n_clusters = len(self.cutoffs) - 1
    self.head_size = self.shortlist_size + self.n_clusters
    self.keep_order = keep_order
    self.out_layers = []
    self.out_projs = []
