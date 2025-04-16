def _calculate_anchors(self, sizes, aspect_ratios, angles):
    cell_anchors = [self.generate_cell_anchors(size, aspect_ratio, angle).
        float() for size, aspect_ratio, angle in zip(sizes, aspect_ratios,
        angles)]
    return BufferList(cell_anchors)
