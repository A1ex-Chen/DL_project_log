def encode_image(self, image_paths):
    """encode images to llama inputs

        :param tupe image_paths: (bsz, )
        :return tensor, tensor: input feature to llama, attention mask to llama
        """
    if self.encoder_pretrain == 'clip':
        inputs = self.load_and_transform_image_data_clip(image_paths, self.
            device)
        inputs = inputs.to(dtype=self.llama_model.dtype, device=self.device)
        inputs_llama = self.clip_encode_image(inputs)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(
            self.device)
        return inputs_llama, atts_llama
    else:
        raise NotImplementedError('Encoder not implemented!')
