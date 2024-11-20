def test_from_save_pretrained_dtype(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        if torch_device == 'mps' and dtype == torch.bfloat16:
            continue
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.to(dtype)
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname,
                low_cpu_mem_usage=True, torch_dtype=dtype)
            assert new_model.dtype == dtype
            new_model = self.model_class.from_pretrained(tmpdirname,
                low_cpu_mem_usage=False, torch_dtype=dtype)
            assert new_model.dtype == dtype
