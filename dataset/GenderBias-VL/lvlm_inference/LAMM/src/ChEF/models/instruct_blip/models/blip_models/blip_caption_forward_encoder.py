def forward_encoder(self, samples):
    image_embeds = self.visual_encoder.forward_features(samples['image'])
    return image_embeds
