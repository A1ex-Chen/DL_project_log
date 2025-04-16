def test_t2i_adapter_sdxl(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
            examples/t2i_adapter/train_t2i_adapter_sdxl.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-xl-pipe
            --adapter_model_name_or_path=hf-internal-testing/tiny-adapter
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=9
            --checkpointing_steps=2
            """
            .split())
        run_command(self._launch_args + test_args)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir,
            'diffusion_pytorch_model.safetensors')))
