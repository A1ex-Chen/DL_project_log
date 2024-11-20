def forward(self, image):
    image_features = self.encode_image(image)
    image_features = F.normalize(image_features, dim=-1)
    return image_features
