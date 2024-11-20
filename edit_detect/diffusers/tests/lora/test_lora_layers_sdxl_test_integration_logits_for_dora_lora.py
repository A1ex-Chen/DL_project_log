@nightly
def test_integration_logits_for_dora_lora(self):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0')
    pipeline.load_lora_weights('hf-internal-testing/dora-trained-on-kohya')
    pipeline.enable_model_cpu_offload()
    images = pipeline('photo of ohwx dog', num_inference_steps=10,
        generator=torch.manual_seed(0), output_type='np').images
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    expected_slice_scale = np.array([0.3932, 0.3742, 0.4429, 0.3737, 0.3504,
        0.433, 0.3948, 0.3769, 0.4516])
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
