def get_clip_image_embeddings(self, image, batch_size):
    clip_image_input = self.feature_extractor.preprocess(image)
    clip_image_features = torch.from_numpy(clip_image_input['pixel_values'][0]
        ).unsqueeze(0).to(self.device).half()
    image_embeddings_clip = self.clip_model.get_image_features(
        clip_image_features)
    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(
        p=2, dim=-1, keepdim=True)
    image_embeddings_clip = image_embeddings_clip.repeat_interleave(batch_size,
        dim=0)
    return image_embeddings_clip
