def train_step(self, data, it=None, epoch=None, scheduling_start=None,
    render_path=None):
    """ Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            epoch(int): current number of epochs
            scheduling_start(int): num of epochs to start scheduling
        """
    self.model.train()
    self.optimizer.zero_grad()
    if self.pose_param_net:
        self.pose_param_net.train()
        self.optimizer_pose.zero_grad()
    if self.focal_net:
        self.focal_net.train()
        self.optimizer_focal.zero_grad()
    if self.distortion_net:
        self.distortion_net.train()
        self.optimizer_distortion.zero_grad()
    loss_dict = self.compute_loss(data, it=it, epoch=epoch,
        scheduling_start=scheduling_start, out_render_path=render_path)
    loss = loss_dict['loss']
    loss.backward()
    self.optimizer.step()
    if self.optimizer_pose:
        self.optimizer_pose.step()
    if self.optimizer_focal:
        self.optimizer_focal.step()
    if self.optimizer_distortion:
        self.optimizer_distortion.step()
    return loss_dict
