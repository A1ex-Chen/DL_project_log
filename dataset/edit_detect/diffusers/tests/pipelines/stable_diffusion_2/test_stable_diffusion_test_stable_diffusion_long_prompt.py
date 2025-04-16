def test_stable_diffusion_long_prompt(self):
    components = self.get_dummy_components()
    components['scheduler'] = LMSDiscreteScheduler.from_config(components[
        'scheduler'].config)
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    do_classifier_free_guidance = True
    negative_prompt = None
    num_images_per_prompt = 1
    logger = logging.get_logger(
        'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion')
    logger.setLevel(logging.WARNING)
    prompt = 25 * '@'
    with CaptureLogger(logger) as cap_logger_3:
        text_embeddings_3, negeative_text_embeddings_3 = sd_pipe.encode_prompt(
            prompt, torch_device, num_images_per_prompt,
            do_classifier_free_guidance, negative_prompt)
        if negeative_text_embeddings_3 is not None:
            text_embeddings_3 = torch.cat([negeative_text_embeddings_3,
                text_embeddings_3])
    prompt = 100 * '@'
    with CaptureLogger(logger) as cap_logger:
        text_embeddings, negative_embeddings = sd_pipe.encode_prompt(prompt,
            torch_device, num_images_per_prompt,
            do_classifier_free_guidance, negative_prompt)
        if negative_embeddings is not None:
            text_embeddings = torch.cat([negative_embeddings, text_embeddings])
    negative_prompt = 'Hello'
    with CaptureLogger(logger) as cap_logger_2:
        text_embeddings_2, negative_text_embeddings_2 = sd_pipe.encode_prompt(
            prompt, torch_device, num_images_per_prompt,
            do_classifier_free_guidance, negative_prompt)
        if negative_text_embeddings_2 is not None:
            text_embeddings_2 = torch.cat([negative_text_embeddings_2,
                text_embeddings_2])
    assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
    assert text_embeddings.shape[1] == 77
    assert cap_logger.out == cap_logger_2.out
    assert cap_logger.out.count('@') == 25
    assert cap_logger_3.out == ''
