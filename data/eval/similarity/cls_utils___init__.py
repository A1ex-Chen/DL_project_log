def __init__(self, input_dim=1536, train_dataset=None, valid_dataset=None,
    test_dataset=None):
    super(ProbingModel, self).__init__()
    self.input_dim = input_dim
    self.linear = nn.Linear(self.input_dim, 256)
    self.linear2 = nn.Linear(256, 1)
    self.output = nn.Sigmoid()
    self.lr = 0.0001
    self.batch_size = 200
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.test_dataset = test_dataset
    self.test_y = []
    self.test_y_hat = []
    self.dropout = nn.Dropout(p=0.19)
