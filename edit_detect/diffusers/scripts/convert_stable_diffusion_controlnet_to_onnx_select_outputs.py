def select_outputs(self, keep, names=None):
    self.graph.outputs = [self.graph.outputs[o] for o in keep]
    if names:
        for i, name in enumerate(names):
            self.graph.outputs[i].name = name
