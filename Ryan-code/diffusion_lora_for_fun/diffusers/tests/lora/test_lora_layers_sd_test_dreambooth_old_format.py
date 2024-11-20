def test_dreambooth_old_format(self):
    generator = torch.Generator('cpu').manual_seed(0)
    lora_model_id = 'hf-internal-testing/lora_dreambooth_dog_example'
    card = RepoCard.load(lora_model_id)
    base_model_id = card.data.to_dict()['base_model']
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id,
        safety_checker=None)
    pipe = pipe.to(torch_device)
    pipe.load_lora_weights(lora_model_id)
    images = pipe('A photo of a sks dog floating in the river', output_type
        ='np', generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.7207, 0.6787, 0.601, 0.7478, 0.6838, 0.6064, 
        0.6984, 0.6443, 0.5785])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.0001
    pipe.unload_lora_weights()
    release_memory(pipe)
