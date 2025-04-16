def move_to_front(self, node):
    if self.front == node:
        return
    if node.next is None:
        node.prev.next = None
        self.back = node.prev
        node.prev = None
    else:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.next = None
        node.prev = None
    self._add_to_front(node)
