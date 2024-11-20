def __init__(self, model, loss, cuda=True, memory_format=torch.
    contiguous_format):
    super(ModelAndLoss, self).__init__()
    if cuda:
        model = model.cuda().to(memory_format=memory_format)
    criterion = loss()
    if cuda:
        criterion = criterion.cuda()
    self.model = model
    self.loss = criterion
