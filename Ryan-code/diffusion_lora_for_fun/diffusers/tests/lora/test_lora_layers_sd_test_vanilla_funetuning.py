def test_vanilla_funetuning(self):
    generator = torch.Generator().manual_seed(0)
    lora_model_id = 'hf-internal-testing/sd-model-finetuned-lora-t4'
    card = RepoCard.load(lora_model_id)
    base_model_id = card.data.to_dict()['base_model']
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id,
        safety_checker=None)
    pipe = pipe.to(torch_device)
    pipe.load_lora_weights(lora_model_id)
    images = pipe('A pokemon with blue eyes.', output_type='np', generator=
        generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.7406, 0.699, 0.5963, 0.7493, 0.7045, 0.6096, 
        0.6886, 0.6388, 0.583])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.0001
    pipe.unload_lora_weights()
    release_memory(pipe)
