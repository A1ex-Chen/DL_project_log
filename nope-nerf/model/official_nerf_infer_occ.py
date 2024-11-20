def infer_occ(self, p):
    pos_enc = encode_position(p, levels=10, inc_input=True)
    x = self.layers0(pos_enc)
    x = torch.cat([x, pos_enc], dim=-1)
    x = self.layers1(x)
    density = self.fc_density(x)
    return x, density
