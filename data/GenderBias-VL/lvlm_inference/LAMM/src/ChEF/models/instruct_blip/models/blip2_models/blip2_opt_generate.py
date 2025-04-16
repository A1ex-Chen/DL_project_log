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
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device)
        if 'prompt' in samples.keys():
            prompt = samples['prompt']
        else:
            prompt = self.prompt
        prompt = [prompt] * image.size(0)
        opt_tokens = self.opt_tokenizer(prompt, return_tensors='pt',
            padding='longest', truncation=True, max_length=self.max_txt_len
            ).to(image.device)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1
            )
        inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.
            input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        outputs = self.opt_model.generate(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, do_sample=use_nucleus_sampling,
            top_p=top_p, temperature=temperature, num_beams=num_beams,
            max_length=max_length, min_length=min_length, eos_token_id=self
            .eos_token_id, repetition_penalty=repetition_penalty,
            length_penalty=length_penalty, num_return_sequences=num_captions)
        output_text = self.opt_tokenizer.batch_decode(outputs,
            skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text
