def get_image_processor(self, repo_id):
    image_processor = CLIPImageProcessor.from_pretrained(repo_id)
    return image_processor
