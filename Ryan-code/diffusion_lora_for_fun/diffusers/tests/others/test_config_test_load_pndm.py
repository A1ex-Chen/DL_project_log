def test_load_pndm(self):
    logger = logging.get_logger('diffusers.configuration_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        pndm = PNDMScheduler.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'scheduler')
    assert pndm.__class__ == PNDMScheduler
    assert cap_logger.out == ''
