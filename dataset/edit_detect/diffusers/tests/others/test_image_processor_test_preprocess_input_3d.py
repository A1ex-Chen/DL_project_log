def test_preprocess_input_3d(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
    input_pt_4d = self.dummy_sample
    input_pt_3d = input_pt_4d.squeeze(0)
    out_pt_4d = image_processor.postprocess(image_processor.preprocess(
        input_pt_4d), output_type='np')
    out_pt_3d = image_processor.postprocess(image_processor.preprocess(
        input_pt_3d), output_type='np')
    input_np_4d = self.to_np(self.dummy_sample)
    input_np_3d = input_np_4d.squeeze(0)
    out_np_4d = image_processor.postprocess(image_processor.preprocess(
        input_np_4d), output_type='np')
    out_np_3d = image_processor.postprocess(image_processor.preprocess(
        input_np_3d), output_type='np')
    assert np.abs(out_pt_4d - out_pt_3d).max() < 1e-06
    assert np.abs(out_np_4d - out_np_3d).max() < 1e-06
