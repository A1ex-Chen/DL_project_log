def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order
    =False):
    super().__init__()
    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.cutoffs = cutoffs + [n_token]
    self.cutoff_ends = [0] + self.cutoffs
    self.div_val = div_val
    self.shortlist_size = self.cutoffs[0]
    self.n_clusters = len(self.cutoffs) - 1
    self.head_size = self.shortlist_size + self.n_clusters
    if self.n_clusters > 0:
        self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters,
            self.d_embed))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
    self.out_layers = nn.ModuleList()
    self.out_projs = nn.ParameterList()
    if div_val == 1:
        for i in range(len(self.cutoffs)):
            if d_proj != d_embed:
                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj,
                    d_embed)))
            else:
                self.out_projs.append(None)
        self.out_layers.append(nn.Linear(d_embed, n_token))
    else:
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = d_embed // div_val ** i
            self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj,
                d_emb_i)))
            self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
    self.keep_order = keep_order
