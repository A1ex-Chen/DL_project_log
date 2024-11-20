def text_prompt(self, text):
    """Processes a text prompt, applies it to existing results and returns the updated results."""
    if self.results[0].masks is not None:
        format_results = self._format_results(self.results[0], 0)
        cropped_boxes, cropped_images, not_crop, filter_id, annotations = (self
            ._crop_image(format_results))
        clip_model, preprocess = self.clip.load('ViT-B/32', device=self.device)
        scores = self.retrieve(clip_model, preprocess, cropped_boxes, text,
            device=self.device)
        max_idx = scores.argsort()
        max_idx = max_idx[-1]
        max_idx += sum(np.array(filter_id) <= int(max_idx))
        self.results[0].masks.data = torch.tensor(np.array([annotations[
            max_idx]['segmentation']]))
    return self.results
