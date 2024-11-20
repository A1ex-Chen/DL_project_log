def _unflat_polygons(x):
    """Unflats/recovers 1-d padded polygons to 3-d polygon list.

  Args:
    x: numpay.array. shape [num_elements, 1], num_elements = num_obj *
      num_vertex + padding.

  Returns:
    A list of three dimensions: [#obj, #polygon, #vertex]
  """
    num_segs = _np_array_split(x, MASK_SEPARATOR)
    polygons = []
    for s in num_segs:
        polygons.append(_np_array_split(s, POLYGON_SEPARATOR))
    polygons = [[polygon.tolist() for polygon in obj] for obj in polygons]
    return polygons
