def eval_step(self, data):
    """ Performs a validation step.

        Args:
            data (tensor): validation data
        """
    self.model.eval()
    device = self.device
    inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
    batch_size, seq_len, n_pts, _ = inputs.size()
    eval_dict = {}
    loss = 0
    with torch.no_grad():
        c_p, c_m, c_i = self.model.encode_inputs(inputs)
        eval_dict_iou = self.eval_step_iou(data, c_m=c_m, c_p=c_p, c_i=c_i)
        for k, v in eval_dict_iou.items():
            eval_dict[k] = v
            loss += eval_dict['iou']
    eval_dict['loss'] = loss.mean().item()
    return eval_dict
