def max_decoder_positions(self):
    return min([m.max_decoder_positions() for m in self.models if hasattr(m,
        'max_decoder_positions')] + [sys.maxsize])
