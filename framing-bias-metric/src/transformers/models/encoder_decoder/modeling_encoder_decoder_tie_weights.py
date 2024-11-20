def tie_weights(self):
    if self.config.tie_encoder_decoder:
        decoder_base_model_prefix = self.decoder.base_model_prefix
        self._tie_encoder_decoder_weights(self.encoder, self.decoder.
            _modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )
