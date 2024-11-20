def _convert_masks(self, masks_or_polygons):
    """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """
    m = masks_or_polygons
    if isinstance(m, PolygonMasks):
        m = m.polygons
    if isinstance(m, BitMasks):
        m = m.tensor.numpy()
    if isinstance(m, torch.Tensor):
        m = m.numpy()
    ret = []
    for x in m:
        if isinstance(x, GenericMask):
            ret.append(x)
        else:
            ret.append(GenericMask(x, self.output.height, self.output.width))
    return ret
