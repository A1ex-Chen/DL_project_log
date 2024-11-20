def initialize_biases(self):
    prior_prob = self.prior_prob
    for conv in self.cls_preds:
        b = conv.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    for conv in self.obj_preds:
        b = conv.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
