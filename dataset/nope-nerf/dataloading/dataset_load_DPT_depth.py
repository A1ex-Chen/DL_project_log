def load_DPT_depth(self, idx, data={}):
    depth_dpt = self.dpt_depth[idx]
    data['dpt'] = depth_dpt
