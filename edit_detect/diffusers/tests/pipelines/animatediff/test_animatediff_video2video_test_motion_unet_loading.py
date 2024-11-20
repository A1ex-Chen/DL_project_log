def test_motion_unet_loading(self):
    components = self.get_dummy_components()
    pipe = AnimateDiffVideoToVideoPipeline(**components)
    assert isinstance(pipe.unet, UNetMotionModel)
