def list_to_array(self, backward=False):
    items = []
    ptr = self.list.front if not backward else self.list.back
    while ptr is not None:
        items.append((ptr.key, ptr.value))
        ptr = ptr.next if not backward else ptr.prev
    return items
