def select_state_func(self, beam_prediction, image_ids):
    bp = []
    for i, image_id in enumerate(image_ids):
        bp.append(beam_prediction[i, self._num_cls[image_id]].unsqueeze(0))
    return torch.cat(bp, dim=0)
