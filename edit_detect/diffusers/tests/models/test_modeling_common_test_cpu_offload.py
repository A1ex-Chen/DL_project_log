@require_torch_gpu
def test_cpu_offload(self):
    config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**config).eval()
    if model._no_split_modules is None:
        return
    model = model.to(torch_device)
    torch.manual_seed(0)
    base_output = model(**inputs_dict)
    model_size = compute_module_sizes(model)['']
    max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]
        ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.cpu().save_pretrained(tmp_dir)
        for max_size in max_gpu_sizes:
            max_memory = {(0): max_size, 'cpu': model_size * 2}
            new_model = self.model_class.from_pretrained(tmp_dir,
                device_map='auto', max_memory=max_memory)
            self.assertSetEqual(set(new_model.hf_device_map.values()), {0,
                'cpu'})
            self.check_device_map_is_respected(new_model, new_model.
                hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)
            self.assertTrue(torch.allclose(base_output[0], new_output[0],
                atol=1e-05))
