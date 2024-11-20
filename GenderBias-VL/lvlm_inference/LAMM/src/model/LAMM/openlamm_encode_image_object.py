def encode_image_object(self, images):
    """encoder loaded image objects"""
    if self.encoder_pretrain == 'clip':
        inputs = transform_vision_data(images, self.device)
        inputs_llama = self.clip_encode_image(inputs)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            self.device)
        return inputs_llama, atts_llama
    else:
        raise NotImplementedError('Encoder pretrain [{}] not implemented'.
            format(self.encoder_pretrain))
