def forward(self, inp, temperature=1.0, hard=False, **kwargs):
    neighborhood, center = self.group_divider(inp)
    logits = self.encoder(neighborhood)
    logits = self.dgcnn_1(logits, center)
    soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)
    sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook)
    feature = self.dgcnn_2(sampled, center)
    coarse, fine = self.decoder(feature)
    with torch.no_grad():
        whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
        whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), 
            -1, 3)
    assert fine.size(2) == self.group_size
    ret = whole_coarse, whole_fine, coarse, fine, neighborhood, logits
    return ret
