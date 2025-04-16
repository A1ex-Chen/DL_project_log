def _grid_anchors(self, grid_sizes):
    anchors = []
    centers = []
    for size, stride, base_anchors in zip(grid_sizes, self.strides, self.
        cell_anchors):
        shift_x, shift_y = _create_grid_offsets(size, stride, self.offset,
            base_anchors.device)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        center = torch.stack((shift_x, shift_y), dim=1)
        anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            ).reshape(-1, 4))
        centers.append(center.view(-1, 2))
    return anchors, centers
