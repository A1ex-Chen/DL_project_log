def test_unconditional_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(
    self):
    with tempfile.TemporaryDirectory() as tmpdir:
        initial_run_args = (
            f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 1
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --checkpointing_steps=2
                """
            .split())
        run_command(self._launch_args + initial_run_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-4', 'checkpoint-6'})
        resume_run_args = (
            f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 2
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 1
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --resume_from_checkpoint=checkpoint-6
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """
            .split())
        run_command(self._launch_args + resume_run_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-10', 'checkpoint-12'})
