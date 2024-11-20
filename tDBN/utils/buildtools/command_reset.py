def reset(self):
    self.state = NodeState.Normal
    self.prev = []
    self.next = []
    for node in self.prev:
        node.reset()
