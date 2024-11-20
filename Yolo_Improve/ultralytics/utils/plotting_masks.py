def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
    """
        Plot masks on image.

        Args:
            masks (tensor): Predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
            retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
        """
    if self.pil:
        self.im = np.asarray(self.im).copy()
    if len(masks) == 0:
        self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    if im_gpu.device != masks.device:
        im_gpu = im_gpu.to(masks.device)
    colors = torch.tensor(colors, device=masks.device, dtype=torch.float32
        ) / 255.0
    colors = colors[:, None, None]
    masks = masks.unsqueeze(3)
    masks_color = masks * (colors * alpha)
    inv_alpha_masks = (1 - masks * alpha).cumprod(0)
    mcs = masks_color.max(dim=0).values
    im_gpu = im_gpu.flip(dims=[0])
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()
    im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
    im_mask = im_gpu * 255
    im_mask_np = im_mask.byte().cpu().numpy()
    self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np,
        self.im.shape)
    if self.pil:
        self.fromarray(self.im)
