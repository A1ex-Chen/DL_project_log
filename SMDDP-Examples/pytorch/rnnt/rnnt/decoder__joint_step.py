def _joint_step(self, model, enc, pred, log_normalize=False):
    logits = model.joint(enc, pred)[:, 0, 0, :]
    if log_normalize:
        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
        return probs
    else:
        return logits
