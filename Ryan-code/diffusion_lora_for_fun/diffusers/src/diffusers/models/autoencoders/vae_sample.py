def sample(self, generator: Optional[torch.Generator]=None) ->torch.Tensor:
    sample = randn_tensor(self.mean.shape, generator=generator, device=self
        .parameters.device, dtype=self.parameters.dtype)
    x = self.mean + self.std * sample
    return x
