def __init__(self, data, dims, translate, scale, axis_order):
    self.data = data
    self.dims = dims
    self.translate = translate
    self.scale = scale
    assert axis_order in ('xzy', 'xyz')
    self.axis_order = axis_order
