def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, dtype=None,
    tie_projs=None, out_layers_weights=None, out_projs=None, keep_order=False):
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
    self.tie_projs = tie_projs
    if self.n_clusters > 0:
        self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters,
            self.d_embed, dtype=dtype, device=torch.device('cuda')))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters, dtype
            =dtype, device=torch.device('cuda')))
    if not out_layers_weights:
        self.out_layers_weights = []
    else:
        self.out_layers_weights = out_layers_weights
    self.out_layers_biases = []
    self.out_projs = []
    if div_val == 1:
        if d_proj != d_embed:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj:
                    self.out_projs.append(out_projs[0])
                else:
                    self.out_projs.append(torch.zeros(d_proj, d_embed,
                        dtype=dtype, device=torch.device('cuda')))
        else:
            for i, tie_proj in enumerate(tie_projs):
                self.out_projs.append(None)
    else:
        for i, tie_proj in enumerate(tie_projs):
            d_emb_i = d_embed // div_val ** i
            if tie_proj:
                self.out_projs.append(out_projs[i])
            else:
                self.out_projs.append(torch.zeros(d_proj, d_emb_i, dtype=
                    dtype, device=torch.device('cuda')))
    if div_val == 1:
        self.out_layers_biases.append(torch.zeros(n_token, dtype=dtype,
            device=torch.device('cuda')))
        if not out_layers_weights:
            self.out_layers_weights.append(nn.Parameter(torch.zeros(n_token,
                d_embed, dtype=dtype, device=torch.device('cuda'))))
    else:
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = d_embed // div_val ** i
            self.out_layers_biases.append(nn.Parameter(torch.zeros(r_idx -
                l_idx, dtype=dtype, device=torch.device('cuda'))))
            if not out_layers_weights:
                self.out_layers_weights.append(nn.Parameter(torch.zeros(
                    r_idx - l_idx, d_emb_i, dtype=dtype, device=torch.
                    device('cuda'))))
    self.keep_order = keep_order
