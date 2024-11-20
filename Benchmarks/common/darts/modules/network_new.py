def new(self):
    """Create a new model initialzed with current alpha parameters.

        Weights are left untouched.

        Returns
        -------
        model : Network
            New model initialized with current alpha.
        """
    model = Network(self.stem, self.cell_dim, self.ops, self.tasks, self.
        criterion).to(self.device)
    for x, y in zip(model.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model
