def test_controlnet_checkpointing_checkpoints_total_limit(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-4', 'checkpoint-6'})
