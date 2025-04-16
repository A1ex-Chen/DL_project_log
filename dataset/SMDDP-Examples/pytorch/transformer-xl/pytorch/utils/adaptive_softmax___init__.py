def __init__(self, in_features, n_classes, cutoffs, keep_order=False):
    super(AdaptiveLogSoftmax, self).__init__()
    cutoffs = list(cutoffs)
    if cutoffs != sorted(cutoffs) or min(cutoffs) <= 0 or max(cutoffs
        ) >= n_classes - 1 or len(set(cutoffs)) != len(cutoffs) or any([(
        int(c) != c) for c in cutoffs]):
        raise ValueError(
            'cutoffs should be a sequence of unique, positive integers sorted in an increasing order, where each value is between 1 and n_classes-1'
            )
    self.in_features = in_features
    self.n_classes = n_classes
    self.cutoffs = cutoffs + [n_classes]
    self.shortlist_size = self.cutoffs[0]
    self.n_clusters = len(self.cutoffs) - 1
    self.head_size = self.shortlist_size + self.n_clusters
    self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.
        in_features))
    self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
    self.keep_order = keep_order
