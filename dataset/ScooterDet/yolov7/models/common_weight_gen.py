def weight_gen(self):
    weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin,
        self.vector[0, :])
    weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum(
        'oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg),
        self.vector[1, :])
    weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum(
        'oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior),
        self.vector[2, :])
    weight_rbr_1x1_kxk_conv1 = None
    if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
        weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.
            id_tensor).squeeze()
    elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
        weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
    else:
        raise NotImplementedError
    weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2
    if self.groups > 1:
        g = self.groups
        t, ig = weight_rbr_1x1_kxk_conv1.size()
        o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
        weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t /
            g), ig)
        weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o /
            g), tg, h, w)
        weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw',
            weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig,
            h, w)
    else:
        weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw',
            weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)
    weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk,
        self.vector[3, :])
    weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.
        weight_rbr_gconv_pw, self.in_channels)
    weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.
        vector[4, :])
    weight = (weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk +
        weight_rbr_pfir + weight_rbr_gconv)
    return weight
