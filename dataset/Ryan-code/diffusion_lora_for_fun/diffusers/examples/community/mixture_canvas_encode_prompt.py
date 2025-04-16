def encode_prompt(self, text_encoder, device):
    """Encodes the previously tokenized prompt for this diffusion region using a given encoder"""
    assert self.tokenized_prompt is not None, ValueError(
        'Prompt in diffusion region must be tokenized before encoding')
    self.encoded_prompt = text_encoder(self.tokenized_prompt.input_ids.to(
        device))[0]
