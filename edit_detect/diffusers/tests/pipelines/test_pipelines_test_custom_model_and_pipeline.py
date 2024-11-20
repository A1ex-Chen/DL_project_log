def test_custom_model_and_pipeline(self):
    pipe = CustomPipeline(encoder=CustomEncoder(), scheduler=DDIMScheduler())
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe.save_pretrained(tmpdirname, safe_serialization=False)
        pipe_new = CustomPipeline.from_pretrained(tmpdirname)
        pipe_new.save_pretrained(tmpdirname)
    conf_1 = dict(pipe.config)
    conf_2 = dict(pipe_new.config)
    del conf_2['_name_or_path']
    assert conf_1 == conf_2
