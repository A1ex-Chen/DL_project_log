def fc_layers(self, cp, tasks):
    """Create fully connnected layers for each task"""
    fc_layers = {}
    for task, dim in tasks.items():
        fc_layers[task] = nn.Linear(cp, dim).to(self.device)
    return fc_layers
