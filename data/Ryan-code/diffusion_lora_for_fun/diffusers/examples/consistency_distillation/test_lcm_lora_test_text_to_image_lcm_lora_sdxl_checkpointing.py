def test_text_to_image_lcm_lora_sdxl_checkpointing(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/consistency_distillation/train_lcm_distill_lora_sdxl.py
                --pretrained_teacher_model hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --lora_rank 4
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --checkpointing_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-4', 'checkpoint-6'})
        test_args = (
            f"""
                examples/consistency_distillation/train_lcm_distill_lora_sdxl.py
                --pretrained_teacher_model hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --lora_rank 4
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 9
                --checkpointing_steps 2
                --resume_from_checkpoint latest
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-4', 'checkpoint-6',
            'checkpoint-8'})
