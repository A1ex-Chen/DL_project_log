@unittest.skipIf(torch_device != 'npu' or not is_torch_npu_available(),
    reason=
    'torch npu flash attention is only available with NPU and `torch_npu` installed'
    )
def test_set_torch_npu_flash_attn_processor_determinism(self):
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
    assert all(type(proc) == AttnProcessorNPU for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output = model(**self.inputs_dict(0))[0]
        else:
            output = model(**inputs_dict)[0]
    model.enable_npu_flash_attention()
    assert all(type(proc) == AttnProcessorNPU for proc in model.
        attn_processors.values())
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            output_2 = model(**self.inputs_dict(0))[0]
        else:
            output_2 = model(**inputs_dict)[0]
    model.set_attn_processor(AttnProcessorNPU())
    assert all(type(proc) == AttnProcessorNPU for proc in model.
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
