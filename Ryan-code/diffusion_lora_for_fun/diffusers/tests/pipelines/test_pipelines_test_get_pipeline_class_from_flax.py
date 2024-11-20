def test_get_pipeline_class_from_flax(self):
    flax_config = {'_class_name': 'FlaxStableDiffusionPipeline'}
    config = {'_class_name': 'StableDiffusionPipeline'}
    assert _get_pipeline_class(DiffusionPipeline, flax_config
        ) == _get_pipeline_class(DiffusionPipeline, config)
