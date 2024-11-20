@smart_inference_mode()
def forward(self, im):
    if im.dtype != torch.float16:
        im = im.half()
    if self.dynamic and im.shape != self.bindings['images'].shape:
        i = self.model.get_binding_index('images')
        self.context.set_binding_shape(i, im.shape)
        self.bindings['images'] = self.bindings['images']._replace(shape=im
            .shape)
        for name in self.output_names:
            i = self.model.get_binding_index(name)
            self.bindings[name].data.resize_(tuple(self.context.
                get_binding_shape(i)))
    s = self.bindings['images'].shape
    assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
    self.binding_addrs['images'] = int(im.data_ptr())
    self.context.execute_v2(list(self.binding_addrs.values()))
    y = [self.bindings[x].data for x in sorted(self.output_names)]
    if isinstance(y, (list, tuple)):
        return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x
            ) for x in y]
    else:
        return self.from_numpy(y)
