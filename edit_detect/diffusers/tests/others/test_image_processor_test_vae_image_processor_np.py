def test_vae_image_processor_np(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
    input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
    for output_type in ['pt', 'np', 'pil']:
        out = image_processor.postprocess(image_processor.preprocess(
            input_np), output_type=output_type)
        out_np = self.to_np(out)
        in_np = (input_np * 255).round() if output_type == 'pil' else input_np
        assert np.abs(in_np - out_np).max(
            ) < 1e-06, f'decoded output does not match input for output_type {output_type}'
