def eval_step(self, data):
    """ Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
    self.model.eval()
    device = self.device
    inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
    eval_dict = {}
    loss = 0
    with torch.no_grad():
        c_s, c_t = self.model.encode_inputs(inputs)
        q_z, q_z_t = self.model.infer_z(inputs, c=c_t, data=data)
        z, z_t = q_z.rsample(), q_z_t.rsample()
        loss_kl_1 = self.compute_kl(q_z).item()
        loss_kl_2 = self.compute_kl(q_z_t).item()
        loss_kl = loss_kl_1 + loss_kl_2
        eval_dict['kl'] = loss_kl
        eval_dict['kl_1'] = loss_kl_1
        eval_dict['kl_2'] = loss_kl_2
        loss += loss_kl
        if self.eval_iou:
            eval_dict_iou = self.eval_step_iou(data, c_s=c_s, c_t=c_t, z=z,
                z_t=z_t)
            for k, v in eval_dict_iou.items():
                eval_dict[k] = v
            loss += eval_dict['rec_error']
        else:
            eval_dict_mesh = self.eval_step_corr_l2(data, c_t=c_t, z_t=z_t)
            for k, v in eval_dict_mesh.items():
                eval_dict[k] = v
            loss += eval_dict['l2']
    eval_dict['loss'] = loss
    return eval_dict
