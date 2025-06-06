def test_gru_is_half(self):
    configs = [(1, False), (2, False), (2, True)]
    for layers, bidir in configs:
        rnn = nn.GRU(input_size=self.h, hidden_size=self.h, num_layers=
            layers, bidirectional=bidir)
        self.run_rnn_test(rnn, layers, bidir)
