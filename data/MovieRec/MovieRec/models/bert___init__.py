def __init__(self, args):
    super().__init__(args)
    self.bert = BERT(args)
    self.out = nn.Linear(self.bert.hidden, args.num_items + 1)
