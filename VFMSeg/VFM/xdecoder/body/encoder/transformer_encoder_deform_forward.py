def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos
    =None, padding_mask=None):
    output = src
    reference_points = self.get_reference_points(spatial_shapes,
        valid_ratios, device=src.device)
    for _, layer in enumerate(self.layers):
        output = layer(output, pos, reference_points, spatial_shapes,
            level_start_index, padding_mask)
    return output
