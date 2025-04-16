def postorder(self):
    ret = []
    for c in self.children:
        ret.extend(c.postorder())
    ret.append(self)
    return ret
