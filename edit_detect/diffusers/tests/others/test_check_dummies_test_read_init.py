def test_read_init(self):
    objects = read_init()
    self.assertIn('torch', objects)
    self.assertIn('torch_and_transformers', objects)
    self.assertIn('flax_and_transformers', objects)
    self.assertIn('torch_and_transformers_and_onnx', objects)
    self.assertIn('UNet2DModel', objects['torch'])
    self.assertIn('FlaxUNet2DConditionModel', objects['flax'])
    self.assertIn('StableDiffusionPipeline', objects['torch_and_transformers'])
    self.assertIn('FlaxStableDiffusionPipeline', objects[
        'flax_and_transformers'])
    self.assertIn('LMSDiscreteScheduler', objects['torch_and_scipy'])
    self.assertIn('OnnxStableDiffusionPipeline', objects[
        'torch_and_transformers_and_onnx'])
