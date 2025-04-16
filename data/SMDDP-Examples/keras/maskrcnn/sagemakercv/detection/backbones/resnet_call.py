def call(self, inputs, training=True, *args, **kwargs):
    """Creation of the model graph."""
    net = self._local_layers['conv2d'](inputs=inputs)
    if self.norm_type == 'batchnorm':
        net = self._local_layers['batchnorm'](inputs=net, training=False)
    elif self.norm_type == 'groupnorm':
        net = self._local_layers['groupnorm'](inputs=net, training=training)
    net = self._local_layers['maxpool2d'](net)
    c2 = self._local_layers['block_1'](inputs=net, training=False)
    c3 = self._local_layers['block_2'](inputs=c2, training=training)
    c4 = self._local_layers['block_3'](inputs=c3, training=training)
    c5 = self._local_layers['block_4'](inputs=c4, training=training)
    return {(2): c2, (3): c3, (4): c4, (5): c5}
