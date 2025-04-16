def _pred_step(self, model, label, hidden, device):
    if label == self._SOS:
        return model.predict(None, hidden, add_sos=False)
    if label > self.blank_idx:
        label -= 1
    label = label_collate([[label]]).to(device)
    return model.predict(label, hidden, add_sos=False)
