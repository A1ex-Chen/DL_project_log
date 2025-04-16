@property
def decoder_cross_attentions(self):
    warnings.warn(
        '`decoder_cross_attentions` is deprecated and will be removed soon. Please use `cross_attentions` instead.'
        , FutureWarning)
    return self.cross_attentions
