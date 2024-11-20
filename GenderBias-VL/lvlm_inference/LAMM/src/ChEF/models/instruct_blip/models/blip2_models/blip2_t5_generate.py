@torch.no_grad()
def generate(self, samples, use_nucleus_sampling=False, num_beams=5,
    max_length=30, min_length=1, top_p=0.9, repetition_penalty=1.0,
    length_penalty=1.0, num_captions=1, temperature=1):
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
        """
    image = samples['image']
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_embeds = image_embeds.float()
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(query_embeds=query_tokens,
        encoder_hidden_states=image_embeds, encoder_attention_mask=
        image_atts, return_dict=True)
    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.
        device)
    if 'prompt' in samples.keys():
        prompt = samples['prompt']
    else:
        prompt = self.prompt
    if isinstance(prompt, str):
        prompt = [prompt] * image.size(0)
    else:
        assert len(prompt) == image.size(0
            ), 'The number of prompts must be equal to the batch size.'
    input_tokens = self.t5_tokenizer(prompt, padding='longest',
        return_tensors='pt').to(image.device)
    encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
    with self.maybe_autocast(dtype=torch.bfloat16):
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        outputs = self.t5_model.generate(inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts, do_sample=use_nucleus_sampling,
            top_p=top_p, temperature=temperature, num_beams=num_beams,
            max_new_tokens=max_length, min_length=min_length,
            repetition_penalty=repetition_penalty, length_penalty=
            length_penalty, num_return_sequences=num_captions)
        output_text = self.t5_tokenizer.batch_decode(outputs,
            skip_special_tokens=True)
    return output_text
