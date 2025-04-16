def __encode_prompt(self, prompt, negative_prompt):
    """
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
    text_input_ids = self.tokenizer(prompt, padding='max_length',
        max_length=self.tokenizer.model_max_length, truncation=True,
        return_tensors='pt').input_ids.type(torch.int32).to(self.torch_device)
    text_input_ids_inp = device_view(text_input_ids)
    text_embeddings = runEngine(self.engine['clip'], {'input_ids':
        text_input_ids_inp}, self.stream)['text_embeddings'].clone()
    uncond_input_ids = self.tokenizer(negative_prompt, padding='max_length',
        max_length=self.tokenizer.model_max_length, truncation=True,
        return_tensors='pt').input_ids.type(torch.int32).to(self.torch_device)
    uncond_input_ids_inp = device_view(uncond_input_ids)
    uncond_embeddings = runEngine(self.engine['clip'], {'input_ids':
        uncond_input_ids_inp}, self.stream)['text_embeddings']
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype
        =torch.float16)
    return text_embeddings
