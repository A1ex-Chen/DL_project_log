def process_box(self, index, box_np, score):
    new_box_np = np.zeros((box_np.shape[0], 8), dtype=np.float32)
    if score.shape[0] > box_np.shape[0]:
        score = score[:box_np.shape[0]]
    box_np[:, 0] /= self._width[index]
    box_np[:, 2] /= self._width[index]
    box_np[:, 1] /= self._height[index]
    box_np[:, 3] /= self._height[index]
    if box_np.shape[0] > 0:
        new_box_np[:, :4] = box_np
        new_box_np[:, 4] = box_np[:, 2] - box_np[:, 0]
        new_box_np[:, 5] = box_np[:, 3] - box_np[:, 1]
        new_box_np[:, 6] = (box_np[:, 2] - box_np[:, 0]) * (box_np[:, 3] -
            box_np[:, 1])
        min_size = min(score.shape[0], box_np.shape[0])
        new_box_np[:min_size, 7] = score[:min_size]
    return new_box_np
