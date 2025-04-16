def test_text_to_image_lora_sdxl(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
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
