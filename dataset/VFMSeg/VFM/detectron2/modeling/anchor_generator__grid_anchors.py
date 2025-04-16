def _grid_anchors(self, grid_sizes):
    anchors = []
    for size, stride, base_anchors in zip(grid_sizes, self.strides, self.
        cell_anchors):
        shift_x, shift_y = _create_grid_offsets(size, stride, self.offset,
            base_anchors.device)
        zeros = torch.zeros_like(shift_x)
        shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)
        anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)
            ).reshape(-1, 5))
    return anchors
