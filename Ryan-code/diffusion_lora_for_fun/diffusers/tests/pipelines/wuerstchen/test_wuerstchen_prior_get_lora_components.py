def get_lora_components(self):
    prior = self.dummy_prior
    prior_lora_config = LoraConfig(r=4, lora_alpha=4, target_modules=[
        'to_q', 'to_k', 'to_v', 'to_out.0'], init_lora_weights=False)
    prior_lora_attn_procs, prior_lora_layers = create_prior_lora_layers(prior)
    lora_components = {'prior_lora_layers': prior_lora_layers,
        'prior_lora_attn_procs': prior_lora_attn_procs}
    return prior, prior_lora_config, lora_components
