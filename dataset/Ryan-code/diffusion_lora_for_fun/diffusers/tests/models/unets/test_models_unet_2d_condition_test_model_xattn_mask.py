@parameterized.expand([[torch.bool], [torch.long], [torch.float]])
def test_model_xattn_mask(self, mask_dtype):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**{**init_dict, 'attention_head_dim': (8, 16),
        'block_out_channels': (16, 32)})
    model.to(torch_device)
    model.eval()
    cond = inputs_dict['encoder_hidden_states']
    with torch.no_grad():
        full_cond_out = model(**inputs_dict).sample
        assert full_cond_out is not None
        keepall_mask = torch.ones(*cond.shape[:-1], device=cond.device,
            dtype=mask_dtype)
        full_cond_keepallmask_out = model(**{**inputs_dict,
            'encoder_attention_mask': keepall_mask}).sample
        assert full_cond_keepallmask_out.allclose(full_cond_out, rtol=1e-05,
            atol=1e-05
            ), "a 'keep all' mask should give the same result as no mask"
        trunc_cond = cond[:, :-1, :]
        trunc_cond_out = model(**{**inputs_dict, 'encoder_hidden_states':
            trunc_cond}).sample
        assert not trunc_cond_out.allclose(full_cond_out, rtol=1e-05, atol=
            1e-05
            ), 'discarding the last token from our cond should change the result'
        batch, tokens, _ = cond.shape
        mask_last = (torch.arange(tokens) < tokens - 1).expand(batch, -1).to(
            cond.device, mask_dtype)
        masked_cond_out = model(**{**inputs_dict, 'encoder_attention_mask':
            mask_last}).sample
        assert masked_cond_out.allclose(trunc_cond_out, rtol=1e-05, atol=1e-05
            ), 'masking the last token from our cond should be equivalent to truncating that token out of the condition'
