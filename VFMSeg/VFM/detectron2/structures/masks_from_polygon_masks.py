@staticmethod
def from_polygon_masks(polygon_masks: Union['PolygonMasks', List[List[np.
    ndarray]]], height: int, width: int) ->'BitMasks':
    """
        Args:
            polygon_masks (list[list[ndarray]] or PolygonMasks)
            height, width (int)
        """
    if isinstance(polygon_masks, PolygonMasks):
        polygon_masks = polygon_masks.polygons
    masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]
    if len(masks):
        return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))
    else:
        return BitMasks(torch.empty(0, height, width, dtype=torch.bool))
