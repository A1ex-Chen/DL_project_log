def __repr__(self):
    text = '%' + self.name + ': ' + self.dtype
    if self.shape is not None:
        text += '[' + ', '.join([str(x) for x in self.shape]) + ']'
    return text
