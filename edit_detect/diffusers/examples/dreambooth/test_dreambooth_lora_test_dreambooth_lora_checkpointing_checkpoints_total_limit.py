def test_dreambooth_lora_checkpointing_checkpoints_total_limit(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
            examples/dreambooth/train_dreambooth_lora.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=prompt
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-4', 'checkpoint-6'})
