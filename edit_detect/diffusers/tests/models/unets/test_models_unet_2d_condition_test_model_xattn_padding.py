@mark.skip(reason=
    "we currently pad mask by target_length tokens (what unclip needs), whereas stable-diffusion's cross-attn needs to instead pad by remaining_length."
    )
def test_model_xattn_padding(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**{**init_dict, 'attention_head_dim': (8, 16)})
    model.to(torch_device)
    model.eval()
    cond = inputs_dict['encoder_hidden_states']
    with torch.no_grad():
        full_cond_out = model(**inputs_dict).sample
        assert full_cond_out is not None
        batch, tokens, _ = cond.shape
        keeplast_mask = (torch.arange(tokens) == tokens - 1).expand(batch, -1
            ).to(cond.device, torch.bool)
        keeplast_out = model(**{**inputs_dict, 'encoder_attention_mask':
            keeplast_mask}).sample
        assert not keeplast_out.allclose(full_cond_out
            ), "a 'keep last token' mask should change the result"
        trunc_mask = torch.zeros(batch, tokens - 1, device=cond.device,
            dtype=torch.bool)
        trunc_mask_out = model(**{**inputs_dict, 'encoder_attention_mask':
            trunc_mask}).sample
        assert trunc_mask_out.allclose(keeplast_out
            ), "a mask with fewer tokens than condition, will be padded with 'keep' tokens. a 'discard-all' mask missing the final token is thus equivalent to a 'keep last' mask."
