def test_textual_inversion_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(
    self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-1', 'checkpoint-2'})
        resume_run_args = (
            f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
                --save_steps 1
                --num_vectors 2
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --resume_from_checkpoint=checkpoint-2
                --checkpoints_total_limit=2
                """
            .split())
        run_command(self._launch_args + resume_run_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-3'})
