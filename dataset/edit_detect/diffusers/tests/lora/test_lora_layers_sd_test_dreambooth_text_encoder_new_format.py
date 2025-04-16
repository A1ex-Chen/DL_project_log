def test_dreambooth_text_encoder_new_format(self):
    generator = torch.Generator().manual_seed(0)
    lora_model_id = 'hf-internal-testing/lora-trained'
    card = RepoCard.load(lora_model_id)
    base_model_id = card.data.to_dict()['base_model']
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id,
        safety_checker=None)
    pipe = pipe.to(torch_device)
    pipe.load_lora_weights(lora_model_id)
    images = pipe('A photo of a sks dog', output_type='np', generator=
        generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.6628, 0.6138, 0.539, 0.6625, 0.613, 0.5463, 
        0.6166, 0.5788, 0.5359])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.0001
    pipe.unload_lora_weights()
    release_memory(pipe)
