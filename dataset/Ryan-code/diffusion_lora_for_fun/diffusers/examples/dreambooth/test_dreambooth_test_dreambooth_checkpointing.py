def test_dreambooth_checkpointing(self):
    instance_prompt = 'photo'
    pretrained_model_name_or_path = (
        'hf-internal-testing/tiny-stable-diffusion-pipe')
    with tempfile.TemporaryDirectory() as tmpdir:
        initial_run_args = (
            f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt {instance_prompt}
                --resolution 64
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
                """
            .split())
        run_command(self._launch_args + initial_run_args)
        pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
        pipe(instance_prompt, num_inference_steps=1)
        self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'checkpoint-2')))
        self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'checkpoint-4')))
        unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder=
            'checkpoint-2/unet')
        pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
            unet=unet, safety_checker=None)
        pipe(instance_prompt, num_inference_steps=1)
        shutil.rmtree(os.path.join(tmpdir, 'checkpoint-2'))
        resume_run_args = (
            f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt {instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """
            .split())
        run_command(self._launch_args + resume_run_args)
        pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
        pipe(instance_prompt, num_inference_steps=1)
        self.assertFalse(os.path.isdir(os.path.join(tmpdir, 'checkpoint-2')))
        self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'checkpoint-4')))
        self.assertTrue(os.path.isdir(os.path.join(tmpdir, 'checkpoint-6')))
