def test_dreambooth_lora_sdxl_with_edm(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --do_edm_style_training
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir,
            'pytorch_lora_weights.safetensors')))
        lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir,
            'pytorch_lora_weights.safetensors'))
        is_lora = all('lora' in k for k in lora_state_dict.keys())
        self.assertTrue(is_lora)
        starts_with_unet = all(key.startswith('unet') for key in
            lora_state_dict.keys())
        self.assertTrue(starts_with_unet)
