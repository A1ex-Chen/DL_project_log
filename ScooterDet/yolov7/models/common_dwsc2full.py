def dwsc2full(self, weight_dw, weight_pw, groups):
    t, ig, h, w = weight_dw.size()
    o, _, _, _ = weight_pw.size()
    tg = int(t / groups)
    i = int(ig * groups)
    weight_dw = weight_dw.view(groups, tg, ig, h, w)
    weight_pw = weight_pw.squeeze().view(o, groups, tg)
    weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
    return weight_dsc.view(o, i, h, w)
