def test_warning_unused_kwargs(self):
    model_id = 'hf-internal-testing/unet-pipeline-dummy'
    logger = logging.get_logger('diffusers.pipelines')
    with tempfile.TemporaryDirectory() as tmpdirname:
        with CaptureLogger(logger) as cap_logger:
            DiffusionPipeline.from_pretrained(model_id, not_used=True,
                cache_dir=tmpdirname, force_download=True)
    assert cap_logger.out.strip().split('\n')[-1
        ] == "Keyword arguments {'not_used': True} are not expected by DDPMPipeline and will be ignored."
