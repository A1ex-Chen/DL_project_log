def _get_decoder_layers(self):
    return getattr_recursive(self, self.decoder_layers_attr_name)
