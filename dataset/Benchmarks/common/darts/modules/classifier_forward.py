def forward(self, x):
    logits = {}
    for task, _ in self.tasks.items():
        logits[task] = self._modules[task](x)
    return logits
