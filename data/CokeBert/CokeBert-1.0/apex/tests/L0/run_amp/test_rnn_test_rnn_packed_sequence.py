def test_rnn_packed_sequence(self):
    num_layers = 2
    rnn = nn.RNN(input_size=self.h, hidden_size=self.h, num_layers=num_layers)
    for typ in [torch.float, torch.half]:
        x = torch.randn((self.t, self.b, self.h), dtype=typ).requires_grad_()
        lens = sorted([random.randint(self.t // 2, self.t) for _ in range(
            self.b)], reverse=True)
        torch.set_default_tensor_type(torch.FloatTensor)
        lens = torch.tensor(lens, dtype=torch.int64, device=torch.device('cpu')
            )
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lens)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        hidden = torch.zeros((num_layers, self.b, self.h), dtype=typ)
        output, _ = rnn(packed_seq, hidden)
        self.assertEqual(output.data.type(), HALF)
        output.data.float().sum().backward()
        self.assertEqual(x.grad.dtype, x.dtype)
