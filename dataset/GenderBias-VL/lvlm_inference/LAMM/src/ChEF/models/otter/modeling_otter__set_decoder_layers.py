def _set_decoder_layers(self, value):
    setattr_recursive(self, self.decoder_layers_attr_name, value)
