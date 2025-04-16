def make_decision(self, probability):
    if float(torch.rand(1)) < probability:
        return True
    else:
        return False
