def _encode_empty_text(self):
    """
        Encode text embedding for empty prompt.
        """
    prompt = ''
    text_inputs = self.tokenizer(prompt, padding='do_not_pad', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
    self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
