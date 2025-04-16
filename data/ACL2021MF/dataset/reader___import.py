def __import(self, data, parent=None):
    assert isinstance(data, dict)
    assert 'parent' not in data
    attrs = dict(data)
    children = attrs.pop('Subcategory', [])
    node = self.nodecls(parent=parent, **attrs)
    for child in children:
        self.__import(child, parent=node)
    return node
