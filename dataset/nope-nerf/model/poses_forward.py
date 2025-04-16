def forward(self, cam_id):
    cam_id = int(cam_id)
    r = self.r[cam_id]
    t = self.t[cam_id]
    c2w = make_c2w(r, t)
    if self.init_c2w is not None:
        c2w = c2w @ self.init_c2w[cam_id]
    return c2w
