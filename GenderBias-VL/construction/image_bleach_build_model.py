def build_model(self):
    args = self.args
    config_file = args.config
    grounded_checkpoint = args.grounded_checkpoint
    grounding_model = self._load_grounding_dino_model(config_file,
        grounded_checkpoint, device=self.device)
    self.grounding_model = grounding_model
    sam_checkpoint = args.sam_checkpoint
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(
        self.device))
    self.sam_predictor = sam_predictor
