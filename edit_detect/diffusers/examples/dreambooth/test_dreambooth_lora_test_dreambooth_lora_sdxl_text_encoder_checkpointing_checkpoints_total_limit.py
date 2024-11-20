def test_dreambooth_lora_sdxl_text_encoder_checkpointing_checkpoints_total_limit(
    self):
    pipeline_path = 'hf-internal-testing/tiny-stable-diffusion-xl-pipe'
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path {pipeline_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --train_text_encoder
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """
            .split())
        run_command(self._launch_args + test_args)
        pipe = DiffusionPipeline.from_pretrained(pipeline_path)
        pipe.load_lora_weights(tmpdir)
        pipe('a prompt', num_inference_steps=2)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-4', 'checkpoint-6'})
