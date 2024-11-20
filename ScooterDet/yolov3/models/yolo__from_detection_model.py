def _from_detection_model(self, model, nc=1000, cutoff=10):
    if isinstance(model, DetectMultiBackend):
        model = model.model
    model.model = model.model[:cutoff]
    m = model.model[-1]
    ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels
    c = Classify(ch, nc)
    c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'
    model.model[-1] = c
    self.model = model.model
    self.stride = model.stride
    self.save = []
    self.nc = nc
