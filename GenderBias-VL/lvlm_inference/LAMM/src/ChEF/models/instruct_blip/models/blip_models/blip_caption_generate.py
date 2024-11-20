def generate(self, samples, use_nucleus_sampling=False, num_beams=3,
    max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0,
    num_captions=1):
    """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> samples = {"image": image}
        >>> captions = model.generate(samples)
        >>> captions
        ['a large statue of a person spraying water from a fountain']
        >>> captions = model.generate(samples, use_nucleus_sampling=True, num_captions=3)
        >>> captions # example output, results may vary due to randomness
        ['singapore showing the view of some building',
        'the singapore harbor in twilight, as the weather is going down',
        'the famous singapore fountain at sunset']
        """
    encoder_out = self.forward_encoder(samples)
    image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)
    prompt = [self.prompt] * image_embeds.size(0)
    prompt = self.tokenizer(prompt, return_tensors='pt').to(self.device)
    prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
    prompt.input_ids = prompt.input_ids[:, :-1]
    decoder_out = self.text_decoder.generate_from_encoder(tokenized_prompt=
        prompt, visual_embeds=image_embeds, sep_token_id=self.tokenizer.
        sep_token_id, pad_token_id=self.tokenizer.pad_token_id,
        use_nucleus_sampling=use_nucleus_sampling, num_beams=num_beams,
        max_length=max_length, min_length=min_length, top_p=top_p,
        repetition_penalty=repetition_penalty)
    outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True
        )
    captions = [output[len(self.prompt):] for output in outputs]
    return captions
