def test_preprocess_input_list(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
    input_pt_4d = self.dummy_sample
    input_pt_list = list(input_pt_4d)
    out_pt_4d = image_processor.postprocess(image_processor.preprocess(
        input_pt_4d), output_type='np')
    out_pt_list = image_processor.postprocess(image_processor.preprocess(
        input_pt_list), output_type='np')
    input_np_4d = self.to_np(self.dummy_sample)
    input_np_list = list(input_np_4d)
    out_np_4d = image_processor.postprocess(image_processor.preprocess(
        input_np_4d), output_type='np')
    out_np_list = image_processor.postprocess(image_processor.preprocess(
        input_np_list), output_type='np')
    assert np.abs(out_pt_4d - out_pt_list).max() < 1e-06
    assert np.abs(out_np_4d - out_np_list).max() < 1e-06
