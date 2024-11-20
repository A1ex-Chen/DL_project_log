def test_from_pipe_optional_components(self):
    image_encoder = self.dummy_image_encoder
    pipe = AutoPipelineForText2Image.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe', image_encoder=
        image_encoder)
    pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    assert pipe.image_encoder is not None
    pipe = AutoPipelineForText2Image.from_pipe(pipe, image_encoder=None)
    assert pipe.image_encoder is None
