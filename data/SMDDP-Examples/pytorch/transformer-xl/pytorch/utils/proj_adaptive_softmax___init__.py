def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, tie_projs=
    None, out_layers_weights=None, out_projs=None, keep_order=False):
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
            self.d_embed))
        self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
    if not out_layers_weights:
        self.out_layers_weights = nn.ParameterList()
    else:
        self.out_layers_weights = out_layers_weights
    self.out_layers_biases = nn.ParameterList()
    self.shared_out_projs = out_projs
    self.out_projs = OptionalParameterList()
    if div_val == 1:
        if d_proj != d_embed:
            for i in range(len(self.cutoffs)):
                if tie_projs[i]:
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(nn.Parameter(torch.zeros(d_proj,
                        d_embed)))
        else:
            self.out_projs.append(None)
        self.out_layers_biases.append(nn.Parameter(torch.zeros(n_token)))
        if not out_layers_weights:
            self.out_layers_weights.append(nn.Parameter(torch.zeros(n_token,
                d_embed)))
    else:
        for i in range(len(self.cutoffs)):
            l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
            d_emb_i = d_embed // div_val ** i
            if tie_projs[i]:
                self.out_projs.append(None)
            else:
                self.out_projs.append(nn.Parameter(torch.zeros(d_proj,
                    d_emb_i)))
            self.out_layers_biases.append(nn.Parameter(torch.zeros(r_idx -
                l_idx)))
            if not out_layers_weights:
                self.out_layers_weights.append(nn.Parameter(torch.zeros(
                    r_idx - l_idx, d_emb_i)))
    self.keep_order = keep_order
