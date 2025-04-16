def test_load_connected_checkpoint_when_specified(self):
    pipeline_prior = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-prior')
    pipeline_prior_connected = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-prior',
        load_connected_pipeline=True)
    assert pipeline_prior.__class__ == pipeline_prior_connected.__class__
    pipeline = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder')
    pipeline_connected = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder',
        load_connected_pipeline=True)
    assert pipeline.__class__ != pipeline_connected.__class__
    assert pipeline.__class__ == KandinskyV22Pipeline
    assert pipeline_connected.__class__ == KandinskyV22CombinedPipeline
    assert set(pipeline_connected.components.keys()) == set([('prior_' + k) for
        k in pipeline_prior.components.keys()] + list(pipeline.components.
        keys()))
