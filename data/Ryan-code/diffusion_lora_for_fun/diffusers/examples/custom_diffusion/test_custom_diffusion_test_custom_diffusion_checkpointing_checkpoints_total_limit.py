def test_custom_diffusion_checkpointing_checkpoints_total_limit(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --no_safe_serialization
            """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-4', 'checkpoint-6'})
