@torch.no_grad()
def generate_caption(self, images):
    """Generates caption for a given image."""
    text = 'a photography of'
    prev_device = self.caption_generator.device
    device = self._execution_device
    inputs = self.caption_processor(images, text, return_tensors='pt').to(
        device=device, dtype=self.caption_generator.dtype)
    self.caption_generator.to(device)
    outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)
    self.caption_generator.to(prev_device)
    caption = self.caption_processor.batch_decode(outputs,
        skip_special_tokens=True)[0]
    return caption
