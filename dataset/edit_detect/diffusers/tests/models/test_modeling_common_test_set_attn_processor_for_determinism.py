@require_torch_gpu
def test_set_attn_processor_for_determinism(self):
    torch.use_deterministic_algorithms(False)
    if self.forward_requires_fresh_args:
        model = self.model_class(**self.init_dict)
    else:
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
    model.to(torch_device)
    if not hasattr(model, 'set_attn_processor'):
        return
    assert all(type(proc) == AttnProcessor2_0 for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_1 = model(**self.inputs_dict(0))[0]
        else:
            output_1 = model(**inputs_dict)[0]
    model.set_default_attn_processor()
    assert all(type(proc) == AttnProcessor for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_2 = model(**self.inputs_dict(0))[0]
        else:
            output_2 = model(**inputs_dict)[0]
    model.set_attn_processor(AttnProcessor2_0())
    assert all(type(proc) == AttnProcessor2_0 for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_4 = model(**self.inputs_dict(0))[0]
        else:
            output_4 = model(**inputs_dict)[0]
    model.set_attn_processor(AttnProcessor())
    assert all(type(proc) == AttnProcessor for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_5 = model(**self.inputs_dict(0))[0]
        else:
            output_5 = model(**inputs_dict)[0]
    torch.use_deterministic_algorithms(True)
    assert torch.allclose(output_2, output_1, atol=self.base_precision)
    assert torch.allclose(output_2, output_4, atol=self.base_precision)
    assert torch.allclose(output_2, output_5, atol=self.base_precision)
