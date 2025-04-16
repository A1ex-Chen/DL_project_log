def test_train_unconditional(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 2
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                """
            .split())
        run_command(self._launch_args + test_args, return_stdout=True)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir, 'unet',
            'diffusion_pytorch_model.safetensors')))
        self.assertTrue(os.path.isfile(os.path.join(tmpdir, 'scheduler',
            'scheduler_config.json')))
