def test_dreambooth_lora_sdxl_text_encoder_custom_captions(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/dreambooth/train_dreambooth_lora_sdxl.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-xl-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --caption_column text
                --instance_prompt photo
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --train_text_encoder
                """
            .split())
        run_command(self._launch_args + test_args)
