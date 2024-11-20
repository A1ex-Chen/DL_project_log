def load_model(self):
    self.image_processor = CLIPImageProcessor.from_pretrained(self.
        vision_tower_name)
    self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True
