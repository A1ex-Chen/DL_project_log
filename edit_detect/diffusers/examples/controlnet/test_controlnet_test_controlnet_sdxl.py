def test_controlnet_sdxl(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
            examples/controlnet/train_controlnet_sdxl.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-xl-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet-sdxl
            --max_train_steps=4
            --checkpointing_steps=2
            """
            .split())
        run_command(self._launch_args + test_args)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir,
            'diffusion_pytorch_model.safetensors')))
