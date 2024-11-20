def __repr__(self):
    text = 'Tensor (name: {}, dtype: {}, shape: {})'.format(self.name, self
        .dtype, self.shape)
    return text
