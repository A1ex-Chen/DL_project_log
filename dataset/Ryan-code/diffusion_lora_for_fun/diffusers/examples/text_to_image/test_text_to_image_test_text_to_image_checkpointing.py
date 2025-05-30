def test_text_to_image_checkpointing(self):
    pretrained_model_name_or_path = (
        'hf-internal-testing/tiny-stable-diffusion-pipe')
    prompt = 'a prompt'
    with tempfile.TemporaryDirectory() as tmpdir:
        initial_run_args = (
            f"""
                examples/text_to_image/train_text_to_image.py
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
                """
            .split())
        run_command(self._launch_args + initial_run_args)
        pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
        pipe(prompt, num_inference_steps=1)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-2', 'checkpoint-4'})
        unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder=
            'checkpoint-2/unet')
        pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
            unet=unet, safety_checker=None)
        pipe(prompt, num_inference_steps=1)
        shutil.rmtree(os.path.join(tmpdir, 'checkpoint-2'))
        resume_run_args = (
            f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """
            .split())
        run_command(self._launch_args + resume_run_args)
        pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
        pipe(prompt, num_inference_steps=1)
        self.assertEqual({x for x in os.listdir(tmpdir) if 'checkpoint' in
            x}, {'checkpoint-4', 'checkpoint-5'})
