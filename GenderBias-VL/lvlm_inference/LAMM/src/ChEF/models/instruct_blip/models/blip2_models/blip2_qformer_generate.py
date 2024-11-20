@torch.no_grad()
def generate(self, samples, use_nucleus_sampling=False, num_beams=3,
    max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
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
    image_embeds = self.ln_vision(self.visual_encoder(image))
    if not use_nucleus_sampling:
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
    else:
        num_beams = 1
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device)
    model_kwargs = {'encoder_hidden_states': image_embeds,
        'encoder_attention_mask': image_atts}
    input_ids = torch.LongTensor(image.size(0), 1).fill_(self.tokenizer.
        bos_token_id).to(image.device)
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    outputs = self.Qformer.generate(input_ids=input_ids, query_embeds=
        query_tokens, max_length=max_length, min_length=min_length,
        num_beams=num_beams, do_sample=use_nucleus_sampling, top_p=top_p,
        eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.
        tokenizer.pad_token_id, **model_kwargs)
    captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return captions
