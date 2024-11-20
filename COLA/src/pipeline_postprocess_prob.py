def postprocess_prob(self, fwd_tuple, bwd_tuple, device, crop=1):
    fwd_pred_list = self.decoding_logits(fwd_tuple)
    bwd_pred_list = self.decoding_logits(bwd_tuple)
    forward_before = torch.tensor([self._extract_token_prob(pred, 'before',
        crop=crop) for pred in fwd_pred_list])
    backward_before = torch.tensor([self._extract_token_prob(pred, 'after',
        crop=crop) for pred in bwd_pred_list])
    forward_after = torch.tensor([self._extract_token_prob(pred, 'after',
        crop=crop) for pred in fwd_pred_list])
    backward_after = torch.tensor([self._extract_token_prob(pred, 'before',
        crop=crop) for pred in bwd_pred_list])
    forward_before, backward_before = forward_before.to(device
        ), backward_before.to(device)
    forward_after, backward_after = forward_after.to(device
        ), backward_after.to(device)
    avg_before = (forward_before + backward_before) / 2
    avg_after = (forward_after + backward_after) / 2
    avg = torch.stack([avg_before, avg_after], dim=-1).cpu().numpy()
    return avg
