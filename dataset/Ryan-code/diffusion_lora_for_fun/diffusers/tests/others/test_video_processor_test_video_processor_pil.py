@parameterized.expand(['list_images', 'list_list_images'])
def test_video_processor_pil(self, input_type):
    video_processor = VideoProcessor(do_resize=False, do_normalize=True)
    input = self.get_dummy_sample(input_type=input_type)
    for output_type in ['pt', 'np', 'pil']:
        out = video_processor.postprocess_video(video_processor.
            preprocess_video(input), output_type=output_type)
        out_np = self.to_np(out)
        input_np = self.to_np(input).astype('float32'
            ) / 255.0 if output_type != 'pil' else self.to_np(input)
        assert np.abs(input_np - out_np).max(
            ) < 1e-06, f'Decoded output does not match input for output_type={output_type!r}'
