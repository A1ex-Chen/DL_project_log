def test_single_file_inference_same_as_pretrained(self):
    model_1 = self.model_class.from_pretrained(self.repo_id).to(torch_device)
    model_2 = self.model_class.from_single_file(self.ckpt_path, config=self
        .repo_id).to(torch_device)
    image = self.get_sd_image(33)
    generator = torch.Generator(torch_device)
    with torch.no_grad():
        sample_1 = model_1(image, generator=generator.manual_seed(0)).sample
        sample_2 = model_2(image, generator=generator.manual_seed(0)).sample
    assert sample_1.shape == sample_2.shape
    output_slice_1 = sample_1.flatten().float().cpu()
    output_slice_2 = sample_2.flatten().float().cpu()
    assert numpy_cosine_similarity_distance(output_slice_1, output_slice_2
        ) < 0.0001
