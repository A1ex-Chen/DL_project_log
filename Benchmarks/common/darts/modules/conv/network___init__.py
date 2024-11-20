def __init__(self, tasks, criterion, device='cpu', hyperparams=
    Hyperparameters()):
    super(ConvNetwork, self).__init__()
    self.tasks = tasks
    self.criterion = criterion
    self.device = device
    self.c = hyperparams.c
    self.num_cells = hyperparams.num_cells
    self.num_nodes = hyperparams.num_nodes
    self.channel_multiplier = hyperparams.channel_multiplier
    c_curr = hyperparams.stem_channel_multiplier * self.c
    self.stem = nn.Sequential(nn.Embedding(num_embeddings=hyperparams.
        num_embeddings, embedding_dim=hyperparams.embedding_dim), nn.Conv1d
        (hyperparams.embedding_dim, c_curr, 3, padding=1, bias=False), nn.
        BatchNorm1d(c_curr)).to(self.device)
    cpp, cp, c_curr = c_curr, c_curr, self.c
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(hyperparams.num_cells):
        if i in [hyperparams.num_cells // 3, 2 * hyperparams.num_cells // 3]:
            c_curr *= 2
            reduction = True
        else:
            reduction = False
        cell = Cell(hyperparams.num_nodes, hyperparams.channel_multiplier,
            cpp, cp, c_curr, reduction, reduction_prev).to(self.device)
        reduction_prev = reduction
        self.cells += [cell]
        cpp, cp = cp, hyperparams.channel_multiplier * c_curr
    self.global_pooling = nn.AdaptiveAvgPool1d(1)
    self.classifier = MultitaskClassifier(cp, tasks)
    k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
    num_ops = len(PRIMITIVES)
    self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
    self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))
    with torch.no_grad():
        self.alpha_normal.mul_(0.001)
        self.alpha_reduce.mul_(0.001)
    self._arch_parameters = [self.alpha_normal, self.alpha_reduce]
