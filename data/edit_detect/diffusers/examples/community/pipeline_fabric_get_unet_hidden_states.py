def get_unet_hidden_states(self, z_all, t, prompt_embd):
    cached_hidden_states = []
    for module in self.unet.modules():
        if isinstance(module, BasicTransformerBlock):

            def new_forward(self, hidden_states, *args, **kwargs):
                cached_hidden_states.append(hidden_states.clone().detach().
                    cpu())
                return self.old_forward(hidden_states, *args, **kwargs)
            module.attn1.old_forward = module.attn1.forward
            module.attn1.forward = new_forward.__get__(module.attn1)
    _ = self.unet(z_all, t, encoder_hidden_states=prompt_embd)
    for module in self.unet.modules():
        if isinstance(module, BasicTransformerBlock):
            module.attn1.forward = module.attn1.old_forward
            del module.attn1.old_forward
    return cached_hidden_states
