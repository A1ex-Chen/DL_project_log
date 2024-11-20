def build_all_layer_point_grids(n_per_side: int, n_layers: int,
    scale_per_layer: int) ->List[np.ndarray]:
    """Generate point grids for all crop layers."""
    return [build_point_grid(int(n_per_side / scale_per_layer ** i)) for i in
        range(n_layers + 1)]
