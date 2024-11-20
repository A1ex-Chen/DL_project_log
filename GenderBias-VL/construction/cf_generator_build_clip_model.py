def build_clip_model(self):
    self.clip_model, self.clip_model_preprocess = clip.load(clip_model,
        device=self.device)
    self.clip_model.eval()
