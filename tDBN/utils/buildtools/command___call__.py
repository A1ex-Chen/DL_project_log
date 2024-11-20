def __call__(self, *nodes):
    for node in nodes:
        self.prev.append(node)
        node.next.append(self)
    return self
