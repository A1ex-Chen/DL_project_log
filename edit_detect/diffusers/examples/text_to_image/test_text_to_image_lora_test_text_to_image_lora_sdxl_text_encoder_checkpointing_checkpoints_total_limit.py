def test_text_to_image_lora_sdxl_text_encoder_checkpointing_checkpoints_total_limit(
    self):
    prompt = 'a prompt'
    pipeline_path = 'hf-internal-testing/tiny-stable-diffusion-xl-pipe'
    with tempfile.TemporaryDirectory() as tmpdir:
        initial_run_args = (
            f"""
                examples/text_to_image/train_text_to_image_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --train_text_encoder
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """
            .split())
        run_command(self._launch_args + initial_run_args)
        pipe = DiffusionPipeline.from_pretrained(pipeline_path)
        pipe.load_lora_weights(tmpdir)
        pipe(prompt, num_inference_steps=1)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-4', 'checkpoint-6'})
