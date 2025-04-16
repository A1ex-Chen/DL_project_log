def get_embeddings(self, prompt_ids: jnp.array, params):
    te_1_inputs = prompt_ids[:, 0, :]
    te_2_inputs = prompt_ids[:, 1, :]
    prompt_embeds = self.text_encoder(te_1_inputs, params=params[
        'text_encoder'], output_hidden_states=True)
    prompt_embeds = prompt_embeds['hidden_states'][-2]
    prompt_embeds_2_out = self.text_encoder_2(te_2_inputs, params=params[
        'text_encoder_2'], output_hidden_states=True)
    prompt_embeds_2 = prompt_embeds_2_out['hidden_states'][-2]
    text_embeds = prompt_embeds_2_out['text_embeds']
    prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)
    return prompt_embeds, text_embeds
