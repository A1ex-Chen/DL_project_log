def test_dreambooth_lora_with_text_encoder(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/dreambooth/train_dreambooth_lora.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
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
                --train_text_encoder
                --output_dir {tmpdir}
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir,
            'pytorch_lora_weights.safetensors')))
        lora_state_dict = safetensors.torch.load_file(os.path.join(tmpdir,
            'pytorch_lora_weights.safetensors'))
        keys = lora_state_dict.keys()
        is_text_encoder_present = any(k.startswith('text_encoder') for k in
            keys)
        self.assertTrue(is_text_encoder_present)
        is_correct_naming = all(k.startswith('unet') or k.startswith(
            'text_encoder') for k in keys)
        self.assertTrue(is_correct_naming)
