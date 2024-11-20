def draw_binary_mask(self, binary_mask, color=None, *, edge_color=None,
    text=None, alpha=0.7, area_threshold=10):
    """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
    if color is None:
        color = random_color(rgb=True, maximum=1)
    color = mplc.to_rgb(color)
    has_valid_segment = False
    binary_mask = binary_mask.astype('uint8')
    mask = GenericMask(binary_mask, self.output.height, self.output.width)
    shape2d = binary_mask.shape[0], binary_mask.shape[1]
    if not mask.has_holes:
        for segment in mask.polygons:
            area = mask_util.area(mask_util.frPyObjects([segment], shape2d[
                0], shape2d[1]))
            if area < (area_threshold or 0):
                continue
            has_valid_segment = True
            segment = segment.reshape(-1, 2)
            self.draw_polygon(segment, color=color, edge_color=edge_color,
                alpha=alpha)
    else:
        rgba = np.zeros(shape2d + (4,), dtype='float32')
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (mask.mask == 1).astype('float32') * alpha
        has_valid_segment = True
        self.output.ax.imshow(rgba, extent=(0, self.output.width, self.
            output.height, 0))
    if text is not None and has_valid_segment:
        lighter_color = self._change_color_brightness(color,
            brightness_factor=0.7)
        self._draw_text_in_mask(binary_mask, text, lighter_color)
    return self.output
