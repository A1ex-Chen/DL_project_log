def test(self, batched_inputs):
    images = self.preprocess_image(batched_inputs)
    results, total_time = self.test_forward(images)
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(results,
        batched_inputs, images.image_sizes):
        height = input_per_image.get('height', image_size[0])
        width = input_per_image.get('width', image_size[1])
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({'instances': r, 'time': total_time})
    return processed_results
