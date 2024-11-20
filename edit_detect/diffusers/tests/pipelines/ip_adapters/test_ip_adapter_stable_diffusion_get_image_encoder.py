def get_image_encoder(self, repo_id, subfolder):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(repo_id,
        subfolder=subfolder, torch_dtype=self.dtype).to(torch_device)
    return image_encoder
