@torch.no_grad()
def __call__(self, prompt: Optional[Union[str, List[str]]]=None, height:
    Optional[int]=None, width: Optional[int]=None, num_images_per_prompt:
    int=1, prior_num_inference_steps: int=25, generator: Optional[torch.
    Generator]=None, prior_latents: Optional[torch.Tensor]=None,
    text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]]=None,
    text_attention_mask: Optional[torch.Tensor]=None, prior_guidance_scale:
    float=4.0, decoder_guidance_scale: float=8.0,
    decoder_num_inference_steps: int=50, decoder_num_images_per_prompt:
    Optional[int]=1, decoder_eta: float=0.0, output_type: Optional[str]=
    'pil', return_dict: bool=True):
    if prompt is not None:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
                )
    else:
        batch_size = text_model_output[0].shape[0]
    device = self._execution_device
    batch_size = batch_size * num_images_per_prompt
    do_classifier_free_guidance = (prior_guidance_scale > 1.0 or 
        decoder_guidance_scale > 1.0)
    text_embeddings, text_encoder_hidden_states, text_mask = (self.
        _encode_prompt(prompt, device, num_images_per_prompt,
        do_classifier_free_guidance, text_model_output, text_attention_mask))
    self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device
        )
    prior_timesteps_tensor = self.prior_scheduler.timesteps
    embedding_dim = self.prior.config.embedding_dim
    prior_latents = self.prepare_latents((batch_size, embedding_dim),
        text_embeddings.dtype, device, generator, prior_latents, self.
        prior_scheduler)
    for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
        latent_model_input = torch.cat([prior_latents] * 2
            ) if do_classifier_free_guidance else prior_latents
        predicted_image_embedding = self.prior(latent_model_input, timestep
            =t, proj_embedding=text_embeddings, encoder_hidden_states=
            text_encoder_hidden_states, attention_mask=text_mask
            ).predicted_image_embedding
        if do_classifier_free_guidance:
            (predicted_image_embedding_uncond, predicted_image_embedding_text
                ) = predicted_image_embedding.chunk(2)
            predicted_image_embedding = (predicted_image_embedding_uncond +
                prior_guidance_scale * (predicted_image_embedding_text -
                predicted_image_embedding_uncond))
        if i + 1 == prior_timesteps_tensor.shape[0]:
            prev_timestep = None
        else:
            prev_timestep = prior_timesteps_tensor[i + 1]
        prior_latents = self.prior_scheduler.step(predicted_image_embedding,
            timestep=t, sample=prior_latents, generator=generator,
            prev_timestep=prev_timestep).prev_sample
    prior_latents = self.prior.post_process_latents(prior_latents)
    image_embeddings = prior_latents
    output = self.decoder_pipe(image=image_embeddings, height=height, width
        =width, num_inference_steps=decoder_num_inference_steps,
        guidance_scale=decoder_guidance_scale, generator=generator,
        output_type=output_type, return_dict=return_dict,
        num_images_per_prompt=decoder_num_images_per_prompt, eta=decoder_eta)
    return output
