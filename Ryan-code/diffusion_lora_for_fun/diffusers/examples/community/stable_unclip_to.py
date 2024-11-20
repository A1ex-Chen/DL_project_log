def to(self, torch_device: Optional[Union[str, torch.device]]=None):
    self.decoder_pipe.to(torch_device)
    super().to(torch_device)
