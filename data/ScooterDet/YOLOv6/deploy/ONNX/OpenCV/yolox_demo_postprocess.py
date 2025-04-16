def demo_postprocess(self, outputs):
    grids = []
    expanded_strides = []
    hsizes = [(self.input_size[0] // stride) for stride in self.strides]
    wsizes = [(self.input_size[1] // stride) for stride in self.strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
        xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs
