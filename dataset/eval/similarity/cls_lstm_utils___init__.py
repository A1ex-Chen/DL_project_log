def __init__(self, input_dim=1536, hidden_dim=115, train_dataset=None,
    valid_dataset=None, test_dataset=None):
    super(ProbingModel, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
        num_layers=1, batch_first=True, bidirectional=False)
    self.linear = nn.Linear(hidden_dim * 1, 256)
    self.linear2 = nn.Linear(256, 1)
    self.output = nn.Sigmoid()
    self.dropout = nn.Dropout(p=0.5)
    self.dropout1 = nn.Dropout(p=0.7)
    self.lr = 0.0001
    self.batch_size = 200
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset
    self.test_dataset = test_dataset
    self.test_y = []
    self.test_y_hat = []
