def test_name_or_path(self):
    model_path = 'hf-internal-testing/tiny-stable-diffusion-torch'
    sd = DiffusionPipeline.from_pretrained(model_path)
    assert sd.name_or_path == model_path
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd.save_pretrained(tmpdirname)
        sd = DiffusionPipeline.from_pretrained(tmpdirname)
        assert sd.name_or_path == tmpdirname
