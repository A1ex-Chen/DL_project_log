def forward(self, samples, is_train=True):
    """
        Forward function for training and evaluation.

        Args:
            samples (dict): a dict of input samples, which contains the following keys:
                - image0 (torch.Tensor): input image 0, shape (batch_size, 3, H, W), default H=384, W=384.
                - image1 (torch.Tensor): input image 1, shape (batch_size, 3, H, W), default H=384, W=384.
                - text_input (list): list of strings, each string is a natural language sentence.
                - label (torch.LongTensor): ground truth label with shape (batch_size,).
            is_train (bool): whether the model is in training mode.
                If True, the model will return the loss;
                If False, the model will return the prediction.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_nlvr", "nlvr")
            >>> samples = {
            ...     "image0": torch.randn(2, 3, 384, 384),
            ...     "image1": torch.randn(2, 3, 384, 384),
            ...     "text_input": ["there is a ferret in tall grass", "there are lips in one of the images"],
            ...     "label": torch.tensor([0, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
        """
    text = samples['text_input']
    text = self.tokenizer(text, padding='longest', return_tensors='pt').to(self
        .device)
    text.input_ids[:, 0] = self.tokenizer.enc_token_id
    targets = samples['label']
    image0 = samples['image0']
    image1 = samples['image1']
    images = torch.cat([image0, image1], dim=0)
    image_embeds = self.visual_encoder.forward_features(images)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self
        .device)
    image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
    encoder_output = self.text_encoder(text.input_ids, attention_mask=text.
        attention_mask, encoder_hidden_states=[image0_embeds, image1_embeds
        ], encoder_attention_mask=[image_atts[:image0_embeds.size(0)],
        image_atts[image0_embeds.size(0):]], return_dict=True)
    prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])
    if is_train:
        loss = F.cross_entropy(prediction, targets)
        return BlipOutput(loss=loss, intermediate_output=
            BlipIntermediateOutput(image_embeds=torch.stack([image0_embeds,
            image1_embeds], dim=0), encoder_output=encoder_output))
    else:
        return {'predictions': prediction, 'targets': targets}
