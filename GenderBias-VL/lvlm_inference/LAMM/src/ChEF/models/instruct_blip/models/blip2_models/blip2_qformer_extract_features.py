@torch.no_grad()
def extract_features(self, samples, mode='multimodal'):
    """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
    image = samples.get('image')
    caption = samples.get('text_input')
    assert mode in ['image', 'text', 'multimodal'
        ], "mode must be one of 'image', 'text', 'multimodal'"
    image_embeds, text_embeds, multimodal_embeds = None, None, None
    image_features, text_features = None, None
    if mode == 'image':
        assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=
            torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0
            ], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts, return_dict=True)
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
    elif mode == 'text':
        assert caption is not None, "text input is None for mode 'text' or 'multimodal'"
        text = self.tokenizer(caption, return_tensors='pt', padding=True).to(
            self.device)
        text_output = self.Qformer.bert(text.input_ids, attention_mask=text
            .attention_mask, return_dict=True)
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
    elif mode == 'multimodal':
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=
            torch.long).to(self.device)
        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0
            ], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device)
        text = self.tokenizer(caption, return_tensors='pt', padding=True).to(
            self.device)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        output = self.Qformer.bert(text.input_ids, query_embeds=
            query_tokens, attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts, return_dict=True)
        multimodal_embeds = output.last_hidden_state[:, :query_tokens.size(
            1), :]
    return BlipOutputFeatures(image_embeds=image_embeds, image_embeds_proj=
        image_features, text_embeds=text_embeds, text_embeds_proj=
        text_features, multimodal_embeds=multimodal_embeds)
