def get_prompt(self, batch_size, te_token, te_encoder, device):
    prompts_tokens = te_token.expand(batch_size, -1).view(batch_size, -1).to(
        device)
    past_key_values = te_encoder(prompts_tokens)
    return past_key_values
