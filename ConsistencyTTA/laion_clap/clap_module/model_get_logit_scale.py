def get_logit_scale(self):
    return self.logit_scale_a.exp(), self.logit_scale_t.exp()
