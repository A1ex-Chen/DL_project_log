def test_custom_diffusion(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/custom_diffusion/train_custom_diffusion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt <new1>
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1.0e-05
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --modifier_token <new1>
                --no_safe_serialization
                --output_dir {tmpdir}
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir,
            'pytorch_custom_diffusion_weights.bin')))
        self.assertTrue(os.path.isfile(os.path.join(tmpdir, '<new1>.bin')))
