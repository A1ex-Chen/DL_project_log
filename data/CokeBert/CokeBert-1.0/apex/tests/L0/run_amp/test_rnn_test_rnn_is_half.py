def test_rnn_is_half(self):
    configs = [(1, False), (2, False), (2, True)]
    for layers, bidir in configs:
        rnn = nn.RNN(input_size=self.h, hidden_size=self.h, num_layers=
            layers, nonlinearity='relu', bidirectional=bidir)
        self.run_rnn_test(rnn, layers, bidir)
