def nms(self, mode=True):
    present = type(self.model[-1]) is NMS
    if mode and not present:
        print('Adding NMS... ')
        m = NMS()
        m.f = -1
        m.i = self.model[-1].i + 1
        self.model.add_module(name='%s' % m.i, module=m)
        self.eval()
    elif not mode and present:
        print('Removing NMS... ')
        self.model = self.model[:-1]
    return self
