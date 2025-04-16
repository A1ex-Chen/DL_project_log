@property
def value(self):
    prec_count = torch.clamp(self.prec_count, min=1.0)
    rec_count = torch.clamp(self.rec_count, min=1.0)
    return (self.prec_total / prec_count).cpu(), (self.rec_total / rec_count
        ).cpu()
