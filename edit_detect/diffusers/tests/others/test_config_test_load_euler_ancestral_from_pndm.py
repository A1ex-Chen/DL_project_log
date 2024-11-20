def test_load_euler_ancestral_from_pndm(self):
    logger = logging.get_logger('diffusers.configuration_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        euler = EulerAncestralDiscreteScheduler.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'scheduler')
    assert euler.__class__ == EulerAncestralDiscreteScheduler
    assert cap_logger.out == ''
