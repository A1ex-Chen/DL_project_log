def test_optional_components(self):
    unet = self.dummy_cond_unet()
    pndm = PNDMScheduler.from_config(
        'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
        'scheduler')
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    orig_sd = StableDiffusionPipeline(unet=unet, scheduler=pndm, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=unet,
        feature_extractor=self.dummy_extractor)
    sd = orig_sd
    assert sd.config.requires_safety_checker is True
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd.save_pretrained(tmpdirname)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname,
            feature_extractor=None, safety_checker=None,
            requires_safety_checker=False)
        assert sd.config.requires_safety_checker is False
        assert sd.config.safety_checker == (None, None)
        assert sd.config.feature_extractor == (None, None)
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd.save_pretrained(tmpdirname)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname)
        assert sd.config.requires_safety_checker is False
        assert sd.config.safety_checker == (None, None)
        assert sd.config.feature_extractor == (None, None)
        orig_sd.save_pretrained(tmpdirname)
        shutil.rmtree(os.path.join(tmpdirname, 'safety_checker'))
        with open(os.path.join(tmpdirname, sd.config_name)) as f:
            config = json.load(f)
            config['safety_checker'] = [None, None]
        with open(os.path.join(tmpdirname, sd.config_name), 'w') as f:
            json.dump(config, f)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname,
            requires_safety_checker=False)
        sd.save_pretrained(tmpdirname)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname)
        assert sd.config.requires_safety_checker is False
        assert sd.config.safety_checker == (None, None)
        assert sd.config.feature_extractor == (None, None)
        with open(os.path.join(tmpdirname, sd.config_name)) as f:
            config = json.load(f)
            del config['safety_checker']
            del config['feature_extractor']
        with open(os.path.join(tmpdirname, sd.config_name), 'w') as f:
            json.dump(config, f)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname)
        assert sd.config.requires_safety_checker is False
        assert sd.config.safety_checker == (None, None)
        assert sd.config.feature_extractor == (None, None)
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd.save_pretrained(tmpdirname)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname,
            feature_extractor=self.dummy_extractor)
        assert sd.config.requires_safety_checker is False
        assert sd.config.safety_checker == (None, None)
        assert sd.config.feature_extractor != (None, None)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname,
            feature_extractor=self.dummy_extractor, safety_checker=unet,
            requires_safety_checker=[True, True])
        assert sd.config.requires_safety_checker == [True, True]
        assert sd.config.safety_checker != (None, None)
        assert sd.config.feature_extractor != (None, None)
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd.save_pretrained(tmpdirname)
        sd = StableDiffusionPipeline.from_pretrained(tmpdirname,
            feature_extractor=self.dummy_extractor)
        assert sd.config.requires_safety_checker == [True, True]
        assert sd.config.safety_checker != (None, None)
        assert sd.config.feature_extractor != (None, None)
