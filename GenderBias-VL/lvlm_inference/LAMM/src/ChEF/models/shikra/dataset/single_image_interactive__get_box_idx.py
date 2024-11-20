def _get_box_idx(self, box):
    assert isinstance(box, (tuple, list)), f'{type(box)}'
    assert isinstance(box[0], (int, float)), f'{type(box[0])}'
    assert len(box) == 4
    box = tuple(box)
    if box not in self.boxes:
        self.boxes.append(box)
        return len(self.boxes) - 1
    else:
        return self.boxes.index(box)
