def forward(self, x: torch.Tensor) ->torch.Tensor:
    """The forward method of the `DecoderTiny` class."""
    x = torch.tanh(x / 3) * 3
    if self.training and self.gradient_checkpointing:

        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        if is_torch_version('>=', '1.11.0'):
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(
                self.layers), x, use_reentrant=False)
        else:
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(
                self.layers), x)
    else:
        x = self.layers(x)
    return x.mul(2).sub(1)
