def __init__(self, fields, n_views=0, mode='train'):
    self.fields = fields
    print(mode, ': ', n_views, ' views')
    self.n_views = n_views
