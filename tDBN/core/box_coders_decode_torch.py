def decode_torch(self, encodings, anchors):
    anchors = anchors[..., [0, 1, 3, 4, 6]]
    ret = box_torch_ops.bev_box_decode(encodings, anchors, self.vec_encode,
        self.linear_dim)
    z_fixed = torch.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.
        dtype, device=ret.device)
    h_fixed = torch.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.
        dtype, device=ret.device)
    return torch.cat([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[
        ..., 4:]], dim=-1)
