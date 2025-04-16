def test_load_ddim_from_pndm(self):
    logger = logging.get_logger('diffusers.configuration_utils')
    logger.setLevel(30)
    with CaptureLogger(logger) as cap_logger:
        ddim = DDIMScheduler.from_pretrained(
            'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
            'scheduler')
    assert ddim.__class__ == DDIMScheduler
    assert cap_logger.out == ''
