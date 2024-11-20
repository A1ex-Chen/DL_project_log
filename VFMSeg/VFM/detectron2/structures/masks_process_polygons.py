def process_polygons(polygons_per_instance: List[Union[torch.Tensor, np.
    ndarray]]) ->List[np.ndarray]:
    if not isinstance(polygons_per_instance, list):
        raise ValueError(
            "Cannot create polygons: Expect a list of polygons per instance. Got '{}' instead."
            .format(type(polygons_per_instance)))
    polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
    for polygon in polygons_per_instance:
        if len(polygon) % 2 != 0 or len(polygon) < 6:
            raise ValueError(
                f'Cannot create a polygon from {len(polygon)} coordinates.')
    return polygons_per_instance
