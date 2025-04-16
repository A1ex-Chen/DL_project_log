def test_vae_image_processor_pil(self):
    image_processor = VaeImageProcessor(do_resize=False, do_normalize=True)
    input_np = self.dummy_sample.cpu().numpy().transpose(0, 2, 3, 1)
    input_pil = image_processor.numpy_to_pil(input_np)
    for output_type in ['pt', 'np', 'pil']:
        out = image_processor.postprocess(image_processor.preprocess(
            input_pil), output_type=output_type)
        for i, o in zip(input_pil, out):
            in_np = np.array(i)
            out_np = self.to_np(out) if output_type == 'pil' else (self.
                to_np(out) * 255).round()
            assert np.abs(in_np - out_np).max(
                ) < 1e-06, f'decoded output does not match input for output_type {output_type}'
