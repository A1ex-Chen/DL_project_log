def __init__(self, *args, **kwargs):
    super(Model2Model, self).__init__(*args, **kwargs)
    self.tie_weights()
