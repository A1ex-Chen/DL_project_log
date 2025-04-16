def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
    encoding = self.encode(input)[0]
    quantized_inputs, vq_loss, encoding_inds = self.vq_layer(encoding)
    latent_shape = quantized_inputs.shape
    loss = self.loss_function(self.decode(quantized_inputs), input, vq_loss)
    return [self.decode(quantized_inputs), encoding_inds, latent_shape, loss]
