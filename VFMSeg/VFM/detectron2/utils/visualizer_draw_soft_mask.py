def draw_soft_mask(self, soft_mask, color=None, *, text=None, alpha=0.5):
    """
        Args:
            soft_mask (ndarray): float array of shape (H, W), each value in [0, 1].
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with mask drawn.
        """
    if color is None:
        color = random_color(rgb=True, maximum=1)
    color = mplc.to_rgb(color)
    shape2d = soft_mask.shape[0], soft_mask.shape[1]
    rgba = np.zeros(shape2d + (4,), dtype='float32')
    rgba[:, :, :3] = color
    rgba[:, :, 3] = soft_mask * alpha
    self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.
        height, 0))
    if text is not None:
        lighter_color = self._change_color_brightness(color,
            brightness_factor=0.7)
        binary_mask = (soft_mask > 0.5).astype('uint8')
        self._draw_text_in_mask(binary_mask, text, lighter_color)
    return self.output
