def tie_weights(self):
    """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
    output_embeddings = self.get_output_embeddings()
    if output_embeddings is not None and self.config.tie_word_embeddings:
        self._tie_or_clone_weights(output_embeddings, self.
            get_input_embeddings())
    if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
        if hasattr(self, self.base_model_prefix):
            self = getattr(self, self.base_model_prefix)
        self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.
            base_model_prefix)
