def set_processor(self, processor: 'AttnProcessor'):
    if hasattr(self, 'processor') and isinstance(self.processor, torch.nn.
        Module) and not isinstance(processor, torch.nn.Module):
        logger.info(
            f'You are removing possibly trained weights of {self.processor} with {processor}'
            )
        self._modules.pop('processor')
    self.processor = processor
