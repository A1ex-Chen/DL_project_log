@torch.no_grad()
def __call__(self, prompt: Union[str, List[List[str]]], num_inference_steps:
    Optional[int]=50, guidance_scale: Optional[float]=7.5, eta: Optional[
    float]=0.0, seed: Optional[int]=None, tile_height: Optional[int]=512,
    tile_width: Optional[int]=512, tile_row_overlap: Optional[int]=256,
    tile_col_overlap: Optional[int]=256, guidance_scale_tiles: Optional[
    List[List[float]]]=None, seed_tiles: Optional[List[List[int]]]=None,
    seed_tiles_mode: Optional[Union[str, List[List[str]]]]='full',
    seed_reroll_regions: Optional[List[Tuple[int, int, int, int, int]]]=
    None, cpu_vae: Optional[bool]=False):
    """
        Function to run the diffusion pipeline with tiling support.

        Args:
            prompt: either a single string (no tiling) or a list of lists with all the prompts to use (one list for each row of tiles). This will also define the tiling structure.
            num_inference_steps: number of diffusions steps.
            guidance_scale: classifier-free guidance.
            seed: general random seed to initialize latents.
            tile_height: height in pixels of each grid tile.
            tile_width: width in pixels of each grid tile.
            tile_row_overlap: number of overlap pixels between tiles in consecutive rows.
            tile_col_overlap: number of overlap pixels between tiles in consecutive columns.
            guidance_scale_tiles: specific weights for classifier-free guidance in each tile.
            guidance_scale_tiles: specific weights for classifier-free guidance in each tile. If None, the value provided in guidance_scale will be used.
            seed_tiles: specific seeds for the initialization latents in each tile. These will override the latents generated for the whole canvas using the standard seed parameter.
            seed_tiles_mode: either "full" "exclusive". If "full", all the latents affected by the tile be overriden. If "exclusive", only the latents that are affected exclusively by this tile (and no other tiles) will be overriden.
            seed_reroll_regions: a list of tuples in the form (start row, end row, start column, end column, seed) defining regions in pixel space for which the latents will be overriden using the given seed. Takes priority over seed_tiles.
            cpu_vae: the decoder from latent space to pixel space can require too mucho GPU RAM for large images. If you find out of memory errors at the end of the generation process, try setting this parameter to True to run the decoder in CPU. Slower, but should run without memory issues.

        Examples:

        Returns:
            A PIL image with the generated image.

        """
    if not isinstance(prompt, list) or not all(isinstance(row, list) for
        row in prompt):
        raise ValueError(
            f'`prompt` has to be a list of lists but is {type(prompt)}')
    grid_rows = len(prompt)
    grid_cols = len(prompt[0])
    if not all(len(row) == grid_cols for row in prompt):
        raise ValueError(
            'All prompt rows must have the same number of prompt columns')
    if not isinstance(seed_tiles_mode, str) and (not isinstance(
        seed_tiles_mode, list) or not all(isinstance(row, list) for row in
        seed_tiles_mode)):
        raise ValueError(
            f'`seed_tiles_mode` has to be a string or list of lists but is {type(prompt)}'
            )
    if isinstance(seed_tiles_mode, str):
        seed_tiles_mode = [[seed_tiles_mode for _ in range(len(row))] for
            row in prompt]
    modes = [mode.value for mode in self.SeedTilesMode]
    if any(mode not in modes for row in seed_tiles_mode for mode in row):
        raise ValueError(f'Seed tiles mode must be one of {modes}')
    if seed_reroll_regions is None:
        seed_reroll_regions = []
    batch_size = 1
    height = tile_height + (grid_rows - 1) * (tile_height - tile_row_overlap)
    width = tile_width + (grid_cols - 1) * (tile_width - tile_col_overlap)
    latents_shape = (batch_size, self.unet.config.in_channels, height // 8,
        width // 8)
    generator = torch.Generator('cuda').manual_seed(seed)
    latents = torch.randn(latents_shape, generator=generator, device=self.
        device)
    if seed_tiles is not None:
        for row in range(grid_rows):
            for col in range(grid_cols):
                if (seed_tile := seed_tiles[row][col]) is not None:
                    mode = seed_tiles_mode[row][col]
                    if mode == self.SeedTilesMode.FULL.value:
                        row_init, row_end, col_init, col_end = (
                            _tile2latent_indices(row, col, tile_width,
                            tile_height, tile_row_overlap, tile_col_overlap))
                    else:
                        row_init, row_end, col_init, col_end = (
                            _tile2latent_exclusive_indices(row, col,
                            tile_width, tile_height, tile_row_overlap,
                            tile_col_overlap, grid_rows, grid_cols))
                    tile_generator = torch.Generator('cuda').manual_seed(
                        seed_tile)
                    tile_shape = latents_shape[0], latents_shape[1
                        ], row_end - row_init, col_end - col_init
                    latents[:, :, row_init:row_end, col_init:col_end
                        ] = torch.randn(tile_shape, generator=
                        tile_generator, device=self.device)
    for row_init, row_end, col_init, col_end, seed_reroll in seed_reroll_regions:
        row_init, row_end, col_init, col_end = _pixel2latent_indices(row_init,
            row_end, col_init, col_end)
        reroll_generator = torch.Generator('cuda').manual_seed(seed_reroll)
        region_shape = latents_shape[0], latents_shape[1
            ], row_end - row_init, col_end - col_init
        latents[:, :, row_init:row_end, col_init:col_end] = torch.randn(
            region_shape, generator=reroll_generator, device=self.device)
    accepts_offset = 'offset' in set(inspect.signature(self.scheduler.
        set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs['offset'] = 1
    self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    if isinstance(self.scheduler, LMSDiscreteScheduler):
        latents = latents * self.scheduler.sigmas[0]
    text_input = [[self.tokenizer(col, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors=
        'pt') for col in row] for row in prompt]
    text_embeddings = [[self.text_encoder(col.input_ids.to(self.device))[0] for
        col in row] for row in text_input]
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        for i in range(grid_rows):
            for j in range(grid_cols):
                max_length = text_input[i][j].input_ids.shape[-1]
                uncond_input = self.tokenizer([''] * batch_size, padding=
                    'max_length', max_length=max_length, return_tensors='pt')
                uncond_embeddings = self.text_encoder(uncond_input.
                    input_ids.to(self.device))[0]
                text_embeddings[i][j] = torch.cat([uncond_embeddings,
                    text_embeddings[i][j]])
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    tile_weights = self._gaussian_weights(tile_width, tile_height, batch_size)
    for i, t in tqdm(enumerate(self.scheduler.timesteps)):
        noise_preds = []
        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                px_row_init, px_row_end, px_col_init, px_col_end = (
                    _tile2latent_indices(row, col, tile_width, tile_height,
                    tile_row_overlap, tile_col_overlap))
                tile_latents = latents[:, :, px_row_init:px_row_end,
                    px_col_init:px_col_end]
                latent_model_input = torch.cat([tile_latents] * 2
                    ) if do_classifier_free_guidance else tile_latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                noise_pred = self.unet(latent_model_input, t,
                    encoder_hidden_states=text_embeddings[row][col])['sample']
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    guidance = (guidance_scale if guidance_scale_tiles is
                        None or guidance_scale_tiles[row][col] is None else
                        guidance_scale_tiles[row][col])
                    noise_pred_tile = noise_pred_uncond + guidance * (
                        noise_pred_text - noise_pred_uncond)
                    noise_preds_row.append(noise_pred_tile)
            noise_preds.append(noise_preds_row)
        noise_pred = torch.zeros(latents.shape, device=self.device)
        contributors = torch.zeros(latents.shape, device=self.device)
        for row in range(grid_rows):
            for col in range(grid_cols):
                px_row_init, px_row_end, px_col_init, px_col_end = (
                    _tile2latent_indices(row, col, tile_width, tile_height,
                    tile_row_overlap, tile_col_overlap))
                noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end
                    ] += noise_preds[row][col] * tile_weights
                contributors[:, :, px_row_init:px_row_end, px_col_init:
                    px_col_end] += tile_weights
        noise_pred /= contributors
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
    image = self.decode_latents(latents, cpu_vae)
    return {'images': image}
