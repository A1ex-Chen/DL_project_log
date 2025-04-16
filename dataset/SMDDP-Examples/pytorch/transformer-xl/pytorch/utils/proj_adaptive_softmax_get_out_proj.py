def get_out_proj(self, i):
    if self.tie_projs[i]:
        if len(self.shared_out_projs) == 0:
            return None
        elif len(self.shared_out_projs) == 1:
            return self.shared_out_projs[0]
        else:
            return self.shared_out_projs[i]
    else:
        return self.out_projs[i]
