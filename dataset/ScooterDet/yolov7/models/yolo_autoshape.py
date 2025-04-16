def autoshape(self):
    print('Adding autoShape... ')
    m = autoShape(self)
    copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'),
        exclude=())
    return m
