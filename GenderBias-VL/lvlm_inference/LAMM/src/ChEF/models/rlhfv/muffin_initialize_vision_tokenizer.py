def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer,
    device, tune_mm_mlp_adapter=False):
    self.model.vision_config.use_im_start_end = mm_use_im_start_end
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    self.resize_token_embeddings(len(tokenizer))
    if mm_use_im_start_end:
        num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        (self.model.vision_config.im_start_token, self.model.vision_config.
            im_end_token) = (tokenizer.convert_tokens_to_ids([
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]))
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim
                =0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        if tune_mm_mlp_adapter:
            self.model.orig_embeds_params = [self.get_input_embeddings().
                weight.data.clone().to(device=device)]
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False
    self.model.vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([
        DEFAULT_IMAGE_PATCH_TOKEN])[0]