def encode_pc(self, pc):
    pc_feat = self.point_encoder(pc)
    pc_embed = pc_feat @ self.pc_projection
    return pc_embed, pc_feat
