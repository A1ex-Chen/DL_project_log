def test_text_to_image_face_id(self):
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, torch_dtype=
        self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter-FaceID', subfolder=None,
        weight_name='ip-adapter-faceid_sd15.bin', image_encoder_folder=None)
    pipeline.set_ip_adapter_scale(0.7)
    inputs = self.get_dummy_inputs()
    id_embeds = load_pt(
        'https://huggingface.co/datasets/fabiorigano/testing-images/resolve/main/ai_face2.ipadpt'
        )[0]
    id_embeds = id_embeds.reshape((2, 1, 1, 512))
    inputs['ip_adapter_image_embeds'] = [id_embeds]
    inputs['ip_adapter_image'] = None
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.32714844, 0.3239746, 0.3466797, 0.31835938,
        0.30004883, 0.3251953, 0.3215332, 0.3552246, 0.3251953])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
