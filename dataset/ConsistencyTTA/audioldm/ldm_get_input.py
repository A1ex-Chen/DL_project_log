@torch.no_grad()
def get_input(self, batch, k, return_first_stage_encode=True,
    return_first_stage_outputs=False, force_c_encode=False, cond_key=None,
    return_original_cond=False, bs=None):
    x = super().get_input(batch, k)
    if bs is not None:
        x = x[:bs]
    x = x.to(self.device)
    if return_first_stage_encode:
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
    else:
        z = None
    if self.model.conditioning_key is not None:
        if cond_key is None:
            cond_key = self.cond_stage_key
        if cond_key != self.first_stage_key:
            if cond_key in ['caption', 'coordinates_bbox']:
                xc = batch[cond_key]
            elif cond_key == 'class_label':
                xc = batch
            else:
                xc = super().get_input(batch, cond_key)
                if type(xc) == torch.Tensor:
                    xc = xc.to(self.device)
        else:
            xc = x
        if not self.cond_stage_trainable or force_c_encode:
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                c = self.get_learned_conditioning(xc.to(self.device))
        else:
            c = xc
        if bs is not None:
            c = c[:bs]
    else:
        c = None
        xc = None
        if self.use_positional_encodings:
            pos_x, pos_y = self.compute_latent_shifts(batch)
            c = {'pos_x': pos_x, 'pos_y': pos_y}
    out = [z, c]
    if return_first_stage_outputs:
        xrec = self.decode_first_stage(z)
        out.extend([x, xrec])
    if return_original_cond:
        out.append(xc)
    return out
