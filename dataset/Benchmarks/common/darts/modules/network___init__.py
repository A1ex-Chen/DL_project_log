def __init__(self, stem: nn.Module, cell_dim: int, classifier_dim: int, ops:
    Dict[str, Callable[[int, int, bool], nn.Module]], tasks: Dict[str, int],
    criterion, device='cpu', hyperparams=Hyperparameters()):
    super(Network, self).__init__()
    self.ops = ops
    self.cell_dim = cell_dim
    self.tasks = tasks
    self.criterion = criterion
    self.device = device
    self.num_cells = hyperparams.num_cells
    self.num_nodes = hyperparams.num_nodes
    self.primitives = list(ops.keys())
    self.stem = stem
    self.channel_multiplier = hyperparams.channel_multiplier
    self.c = hyperparams.c
    c_curr = cell_dim * self.channel_multiplier * hyperparams.c
    cpp, cp, c_curr = c_curr, c_curr, hyperparams.c
    self.cells = nn.ModuleList()
    for i in range(hyperparams.num_cells):
        cell = Cell(hyperparams.num_nodes, hyperparams.channel_multiplier,
            cpp, cp, c_curr, self.primitives, self.ops).to(self.device)
        self.cells += [cell]
    self.classifier = MultitaskClassifier(classifier_dim, tasks)
    k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
    num_ops = len(self.primitives)
    self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
    with torch.no_grad():
        self.alpha_normal.mul_(0.001)
    self._arch_parameters = [self.alpha_normal]
