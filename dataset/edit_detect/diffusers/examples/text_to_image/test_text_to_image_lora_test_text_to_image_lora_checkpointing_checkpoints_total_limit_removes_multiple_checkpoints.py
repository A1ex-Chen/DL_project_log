def test_text_to_image_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(
    self):
    pretrained_model_name_or_path = (
        'hf-internal-testing/tiny-stable-diffusion-pipe')
    prompt = 'a prompt'
    with tempfile.TemporaryDirectory() as tmpdir:
        initial_run_args = (
            f"""
                examples/text_to_image/train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                --num_validation_images=0
                """
            .split())
        run_command(self._launch_args + initial_run_args)
        pipe = DiffusionPipeline.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            safety_checker=None)
        pipe.load_lora_weights(tmpdir)
        pipe(prompt, num_inference_steps=1)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-4'})
        resume_run_args = (
            f"""
                examples/text_to_image/train_text_to_image_lora.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 8
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --checkpoints_total_limit=2
                --seed=0
                --num_validation_images=0
                """
            .split())
        run_command(self._launch_args + resume_run_args)
        pipe = DiffusionPipeline.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-pipe',
            safety_checker=None)
        pipe.load_lora_weights(tmpdir)
        pipe(prompt, num_inference_steps=1)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-6', 'checkpoint-8'})
