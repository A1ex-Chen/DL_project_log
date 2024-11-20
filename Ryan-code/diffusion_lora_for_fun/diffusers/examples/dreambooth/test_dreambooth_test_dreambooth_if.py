def test_dreambooth_if(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = (
            f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-if-pipe
                --instance_data_dir docs/source/en/imgs
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
                --pre_compute_text_embeddings
                --tokenizer_max_length=77
                --text_encoder_use_attention_mask
                """
            .split())
        run_command(self._launch_args + test_args)
        self.assertTrue(os.path.isfile(os.path.join(tmpdir, 'unet',
            'diffusion_pytorch_model.safetensors')))
        self.assertTrue(os.path.isfile(os.path.join(tmpdir, 'scheduler',
            'scheduler_config.json')))
