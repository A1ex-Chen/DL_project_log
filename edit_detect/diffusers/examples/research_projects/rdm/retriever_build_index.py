def build_index(self, model=None, feature_extractor: CLIPFeatureExtractor=
    None, torch_dtype=torch.float32):
    if not self.index_initialized:
        model = model or CLIPModel.from_pretrained(self.config.
            clip_name_or_path).to(dtype=torch_dtype)
        feature_extractor = (feature_extractor or CLIPFeatureExtractor.
            from_pretrained(self.config.clip_name_or_path))
        self.dataset = get_dataset_with_emb_from_clip_model(self.dataset,
            model, feature_extractor, image_column=self.config.image_column,
            index_name=self.config.index_name)
        self.init_index()
