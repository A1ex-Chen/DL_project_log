def train_step(self, data, it=100000):
    """ Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        """
    self.model.eval()
    self.pose_param_net.train()
    self.optimizer_pose.zero_grad()
    if self.focal_net is not None:
        self.focal_net.eval()
    loss_dict = self.compute_loss(data, it=it)
    loss = loss_dict['loss']
    loss.backward()
    self.optimizer_pose.step()
    return loss_dict
