def test_preprocess_input_mask_3d(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=False,
        do_binarize=True, do_convert_grayscale=True)
    input_pt_4d = self.dummy_mask
    input_pt_3d = input_pt_4d.squeeze(0)
    input_pt_2d = input_pt_3d.squeeze(0)
    out_pt_4d = image_processor.postprocess(image_processor.preprocess(
        input_pt_4d), output_type='np')
    out_pt_3d = image_processor.postprocess(image_processor.preprocess(
        input_pt_3d), output_type='np')
    out_pt_2d = image_processor.postprocess(image_processor.preprocess(
        input_pt_2d), output_type='np')
    input_np_4d = self.to_np(self.dummy_mask)
    input_np_3d = input_np_4d.squeeze(0)
    input_np_3d_1 = input_np_4d.squeeze(-1)
    input_np_2d = input_np_3d.squeeze(-1)
    out_np_4d = image_processor.postprocess(image_processor.preprocess(
        input_np_4d), output_type='np')
    out_np_3d = image_processor.postprocess(image_processor.preprocess(
        input_np_3d), output_type='np')
    out_np_3d_1 = image_processor.postprocess(image_processor.preprocess(
        input_np_3d_1), output_type='np')
    out_np_2d = image_processor.postprocess(image_processor.preprocess(
        input_np_2d), output_type='np')
    assert np.abs(out_pt_4d - out_pt_3d).max() == 0
    assert np.abs(out_pt_4d - out_pt_2d).max() == 0
    assert np.abs(out_np_4d - out_np_3d).max() == 0
    assert np.abs(out_np_4d - out_np_3d_1).max() == 0
    assert np.abs(out_np_4d - out_np_2d).max() == 0
