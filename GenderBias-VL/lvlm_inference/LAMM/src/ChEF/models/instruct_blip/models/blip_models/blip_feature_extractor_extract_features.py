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

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
    image = samples.get('image')
    caption = samples.get('text_input')
    assert mode in ['image', 'text', 'multimodal'
        ], "mode must be one of 'image', 'text', 'multimodal'"
    image_embeds, text_embeds, multimodal_embeds = None, None, None
    image_features, text_features = None, None
    if mode == 'image':
        assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
        image_embeds = self.visual_encoder.forward_features(image)
        image_features = self.vision_proj(image_embeds)
        image_features = F.normalize(image_features, dim=-1)
    elif mode == 'text':
        assert caption is not None, "text input is None for mode 'text' or 'multimodal'"
        text = self.tokenizer(caption, return_tensors='pt', padding=True).to(
            self.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text
            .attention_mask, return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
    elif mode == 'multimodal':
        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device)
        text = self.tokenizer(caption, return_tensors='pt', padding=True).to(
            self.device)
        text.input_ids[:, 0] = self.tokenizer.enc_token_id
        output = self.text_encoder(text.input_ids, attention_mask=text.
            attention_mask, encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts, return_dict=True)
        multimodal_embeds = output.last_hidden_state
    return BlipOutputFeatures(image_embeds=image_embeds, image_embeds_proj=
        image_features, text_embeds=text_embeds, text_embeds_proj=
        text_features, multimodal_embeds=multimodal_embeds)
