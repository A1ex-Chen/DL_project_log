def initialize_biases(self):
    for conv in self.cls_preds:
        b = conv.bias.view(-1)
        b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
        conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = conv.weight
        w.data.fill_(0.0)
        conv.weight = torch.nn.Parameter(w, requires_grad=True)
    for conv in self.reg_preds:
        b = conv.bias.view(-1)
        b.data.fill_(1.0)
        conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = conv.weight
        w.data.fill_(0.0)
        conv.weight = torch.nn.Parameter(w, requires_grad=True)
