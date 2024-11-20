@torch.no_grad()
def forward(self, images):
    if type(images) is list:
        image_features = []
        for image in images:
            image_forward_out = self.vision_tower(image.to(device=self.
                device, dtype=self.dtype).unsqueeze(0),
                output_hidden_states=True)
            image_feature = self.feature_select(image_forward_out).to(image
                .dtype)
            image_features.append(image_feature)
    else:
        image_forward_outs = self.vision_tower(images.to(device=self.device,
            dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.
            dtype)
    return image_features
