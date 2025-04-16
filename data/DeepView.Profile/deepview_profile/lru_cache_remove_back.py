def remove_back(self):
    if self.size == 0:
        return None
    node = self.back
    if self.size == 1:
        self.front = None
        self.back = None
    else:
        node.prev.next = None
        self.back = node.prev
    self.size -= 1
    return node
