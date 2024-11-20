def test_preprocess_input_mask_3d_batch(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=False,
        do_convert_grayscale=True)
    dummy_mask_batch = torch.cat([self.dummy_mask] * 2, axis=0)
    input_pt_3d = dummy_mask_batch.squeeze(1)
    input_np_3d = self.to_np(dummy_mask_batch).squeeze(-1)
    input_pt_3d_list = list(input_pt_3d)
    input_np_3d_list = list(input_np_3d)
    out_pt_3d = image_processor.postprocess(image_processor.preprocess(
        input_pt_3d), output_type='np')
    out_pt_3d_list = image_processor.postprocess(image_processor.preprocess
        (input_pt_3d_list), output_type='np')
    assert np.abs(out_pt_3d - out_pt_3d_list).max() < 1e-06
    out_np_3d = image_processor.postprocess(image_processor.preprocess(
        input_np_3d), output_type='np')
    out_np_3d_list = image_processor.postprocess(image_processor.preprocess
        (input_np_3d_list), output_type='np')
    assert np.abs(out_np_3d - out_np_3d_list).max() < 1e-06
