@torch.no_grad()
def __call__(self, input_image: Image, denoising_steps: int=10,
    ensemble_size: int=10, processing_res: int=768, match_input_res: bool=
    True, resample_method: str='bilinear', batch_size: int=0, seed: Union[
    int, None]=None, color_map: str='Spectral', show_progress_bar: bool=
    True, ensemble_kwargs: Dict=None) ->MarigoldDepthOutput:
    """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            seed (`int`, *optional*, defaults to `None`)
                Reproducibility seed.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
    device = self.device
    input_size = input_image.size
    if not match_input_res:
        assert processing_res is not None, 'Value error: `resize_output_back` is only valid with '
    assert processing_res >= 0
    assert ensemble_size >= 1
    self._check_inference_step(denoising_steps)
    resample_method: Resampling = get_pil_resample_method(resample_method)
    if processing_res > 0:
        input_image = self.resize_max_res(input_image, max_edge_resolution=
            processing_res, resample_method=resample_method)
    input_image = input_image.convert('RGB')
    image = np.asarray(input_image)
    rgb = np.transpose(image, (2, 0, 1))
    rgb_norm = rgb / 255.0 * 2.0 - 1.0
    rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
    rgb_norm = rgb_norm.to(device)
    assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
    duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
    single_rgb_dataset = TensorDataset(duplicated_rgb)
    if batch_size > 0:
        _bs = batch_size
    else:
        _bs = self._find_batch_size(ensemble_size=ensemble_size, input_res=
            max(rgb_norm.shape[1:]), dtype=self.dtype)
    single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs,
        shuffle=False)
    depth_pred_ls = []
    if show_progress_bar:
        iterable = tqdm(single_rgb_loader, desc=' ' * 2 +
            'Inference batches', leave=False)
    else:
        iterable = single_rgb_loader
    for batch in iterable:
        batched_img, = batch
        depth_pred_raw = self.single_infer(rgb_in=batched_img,
            num_inference_steps=denoising_steps, show_pbar=
            show_progress_bar, seed=seed)
        depth_pred_ls.append(depth_pred_raw.detach())
    depth_preds = torch.concat(depth_pred_ls, dim=0).squeeze()
    torch.cuda.empty_cache()
    if ensemble_size > 1:
        depth_pred, pred_uncert = self.ensemble_depths(depth_preds, **
            ensemble_kwargs or {})
    else:
        depth_pred = depth_preds
        pred_uncert = None
    min_d = torch.min(depth_pred)
    max_d = torch.max(depth_pred)
    depth_pred = (depth_pred - min_d) / (max_d - min_d)
    depth_pred = depth_pred.cpu().numpy().astype(np.float32)
    if match_input_res:
        pred_img = Image.fromarray(depth_pred)
        pred_img = pred_img.resize(input_size, resample=resample_method)
        depth_pred = np.asarray(pred_img)
    depth_pred = depth_pred.clip(0, 1)
    if color_map is not None:
        depth_colored = self.colorize_depth_maps(depth_pred, 0, 1, cmap=
            color_map).squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = self.chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
    else:
        depth_colored_img = None
    return MarigoldDepthOutput(depth_np=depth_pred, depth_colored=
        depth_colored_img, uncertainty=pred_uncert)
