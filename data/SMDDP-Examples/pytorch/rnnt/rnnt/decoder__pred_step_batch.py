def _pred_step_batch(self, label, hidden, device):
    return self.model.predict_batch(label, hidden, add_sos=False)
