def test_overwrite_config_on_load(self):
    logger = logging.get_logger('diffusers.configuration_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        ddpm = DDPMScheduler.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'scheduler', prediction_type='sample', beta_end=8)
    with CaptureLogger(logger) as cap_logger_2:
        ddpm_2 = DDPMScheduler.from_pretrained('google/ddpm-celebahq-256',
            beta_start=88)
    assert ddpm.__class__ == DDPMScheduler
    assert ddpm.config.prediction_type == 'sample'
    assert ddpm.config.beta_end == 8
    assert ddpm_2.config.beta_start == 88
    assert cap_logger.out == ''
    assert cap_logger_2.out == ''
