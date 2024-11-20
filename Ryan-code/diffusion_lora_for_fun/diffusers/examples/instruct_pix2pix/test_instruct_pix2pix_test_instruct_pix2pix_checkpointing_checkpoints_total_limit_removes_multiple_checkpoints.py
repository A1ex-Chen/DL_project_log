def test_instruct_pix2pix_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(
    self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/instruct_pix2pix/train_instruct_pix2pix.py
                --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name=hf-internal-testing/instructpix2pix-10-samples
                --resolution=64
                --random_flip
                --train_batch_size=1
                --max_train_steps=4
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --seed=0
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-4'})
        resume_run_args = (
            f"""
                examples/instruct_pix2pix/train_instruct_pix2pix.py
                --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name=hf-internal-testing/instructpix2pix-10-samples
                --resolution=64
                --random_flip
                --train_batch_size=1
                --max_train_steps=8
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --seed=0
                --resume_from_checkpoint=checkpoint-4
                --checkpoints_total_limit=2
                """
            .split())
        run_command(self._launch_args + resume_run_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-6', 'checkpoint-8'})
