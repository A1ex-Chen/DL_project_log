@unittest.skipIf(torch_device != 'cuda' or not is_xformers_available(),
    reason=
    'XFormers attention is only available with CUDA and `xformers` installed')
def test_set_xformers_attn_processor_for_determinism(self):
    torch.use_deterministic_algorithms(False)
    if self.forward_requires_fresh_args:
        model = self.model_class(**self.init_dict)
    else:
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
    model.to(torch_device)
    if not hasattr(model, 'set_attn_processor'):
        return
    model.set_default_attn_processor()
    assert all(type(proc) == AttnProcessor for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output = model(**self.inputs_dict(0))[0]
        else:
            output = model(**inputs_dict)[0]
    model.enable_xformers_memory_efficient_attention()
    assert all(type(proc) == XFormersAttnProcessor for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_2 = model(**self.inputs_dict(0))[0]
        else:
            output_2 = model(**inputs_dict)[0]
    model.set_attn_processor(XFormersAttnProcessor())
    assert all(type(proc) == XFormersAttnProcessor for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_3 = model(**self.inputs_dict(0))[0]
        else:
            output_3 = model(**inputs_dict)[0]
    torch.use_deterministic_algorithms(True)
    assert torch.allclose(output, output_2, atol=self.base_precision)
    assert torch.allclose(output, output_3, atol=self.base_precision)
    assert torch.allclose(output_2, output_3, atol=self.base_precision)
