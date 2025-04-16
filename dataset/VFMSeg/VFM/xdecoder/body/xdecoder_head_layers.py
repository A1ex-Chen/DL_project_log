def layers(self, features, mask=None, target_queries=None, target_vlp=None,
    task='seg', extra={}):
    mask_features, transformer_encoder_features, multi_scale_features = (self
        .pixel_decoder.forward_features(features))
    if self.transformer_in_feature == 'multi_scale_pixel_decoder':
        predictions = self.predictor(multi_scale_features, mask_features,
            mask, target_queries, target_vlp, task, extra)
    elif self.transformer_in_feature == 'transformer_encoder':
        assert transformer_encoder_features is not None, 'Please use the TransformerEncoderPixelDecoder.'
        predictions = self.predictor(transformer_encoder_features,
            mask_features, mask)
    elif self.transformer_in_feature == 'pixel_embedding':
        predictions = self.predictor(mask_features, mask_features, mask)
    else:
        predictions = self.predictor(features[self.transformer_in_feature],
            mask_features, mask)
    return predictions
