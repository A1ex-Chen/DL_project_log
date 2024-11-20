def _break_up_pc(self, pc):
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1
        ) > 3 else None
    return xyz, features
