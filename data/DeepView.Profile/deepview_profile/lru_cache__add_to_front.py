def _add_to_front(self, node):
    if self.size == 0:
        self.front = node
        self.back = node
    else:
        node.next = self.front
        self.front.prev = node
        self.front = node
