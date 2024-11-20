def test_save_pipeline_change_config(self):
    pipe = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe.save_pretrained(tmpdirname)
        pipe = DiffusionPipeline.from_pretrained(tmpdirname)
        assert pipe.scheduler.__class__.__name__ == 'PNDMScheduler'
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.
            scheduler.config)
        pipe.save_pretrained(tmpdirname)
        pipe = DiffusionPipeline.from_pretrained(tmpdirname)
        assert pipe.scheduler.__class__.__name__ == 'DPMSolverMultistepScheduler'
