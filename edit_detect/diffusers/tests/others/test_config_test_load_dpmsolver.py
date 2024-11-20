def test_load_dpmsolver(self):
    logger = logging.get_logger('diffusers.configuration_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        dpm = DPMSolverMultistepScheduler.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'scheduler')
    assert dpm.__class__ == DPMSolverMultistepScheduler
    assert cap_logger.out == ''
