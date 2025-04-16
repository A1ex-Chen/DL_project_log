def test_preprocess_input_mask_list(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=False,
        do_convert_grayscale=True)
    input_pt_4d = self.dummy_mask
    input_pt_3d = input_pt_4d.squeeze(0)
    input_pt_2d = input_pt_3d.squeeze(0)
    inputs_pt = [input_pt_4d, input_pt_3d, input_pt_2d]
    inputs_pt_list = [[input_pt] for input_pt in inputs_pt]
    for input_pt, input_pt_list in zip(inputs_pt, inputs_pt_list):
        out_pt = image_processor.postprocess(image_processor.preprocess(
            input_pt), output_type='np')
        out_pt_list = image_processor.postprocess(image_processor.
            preprocess(input_pt_list), output_type='np')
        assert np.abs(out_pt - out_pt_list).max() < 1e-06
    input_np_4d = self.to_np(self.dummy_mask)
    input_np_3d = input_np_4d.squeeze(0)
    input_np_2d = input_np_3d.squeeze(-1)
    inputs_np = [input_np_4d, input_np_3d, input_np_2d]
    inputs_np_list = [[input_np] for input_np in inputs_np]
    for input_np, input_np_list in zip(inputs_np, inputs_np_list):
        out_np = image_processor.postprocess(image_processor.preprocess(
            input_np), output_type='np')
        out_np_list = image_processor.postprocess(image_processor.
            preprocess(input_np_list), output_type='np')
        assert np.abs(out_np - out_np_list).max() < 1e-06
