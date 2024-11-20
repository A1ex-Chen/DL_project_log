def compute_predicted_angle(self, angle_logits, angle_residual):
    if angle_logits.shape[-1] == 1:
        angle = angle_logits * 0 + angle_residual * 0
        angle = angle.squeeze(-1).clamp(min=0)
    else:
        angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
        pred_angle_class = angle_logits.argmax(dim=-1).detach()
        angle_center = angle_per_cls * pred_angle_class
        angle = angle_center + angle_residual.gather(2, pred_angle_class.
            unsqueeze(-1)).squeeze(-1)
        mask = angle > np.pi
        angle[mask] = angle[mask] - 2 * np.pi
    return angle
