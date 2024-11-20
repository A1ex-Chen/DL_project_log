def _encode_image(self, retrieved_images, batch_size):
    if len(retrieved_images[0]) == 0:
        return None
    for i in range(len(retrieved_images)):
        retrieved_images[i] = normalize_images(retrieved_images[i])
        retrieved_images[i] = preprocess_images(retrieved_images[i], self.
            feature_extractor).to(self.clip.device, dtype=self.clip.dtype)
    _, c, h, w = retrieved_images[0].shape
    retrieved_images = torch.reshape(torch.cat(retrieved_images, dim=0), (-
        1, c, h, w))
    image_embeddings = self.clip.get_image_features(retrieved_images)
    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings,
        dim=-1, keepdim=True)
    _, d = image_embeddings.shape
    image_embeddings = torch.reshape(image_embeddings, (batch_size, -1, d))
    return image_embeddings
