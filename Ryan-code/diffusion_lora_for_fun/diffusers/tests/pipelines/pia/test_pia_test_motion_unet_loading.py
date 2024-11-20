def test_motion_unet_loading(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    assert isinstance(pipe.unet, UNetMotionModel)
