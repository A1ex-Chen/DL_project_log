def forward(self, x):
    x = x.view(x.size(0), -1, self.input_dim)
    lstm_out, _ = self.lstm(x)
    lstm_out = lstm_out[:, -1, :]
    lstm_out = self.dropout1(lstm_out)
    x1 = self.linear(lstm_out)
    x1 = self.dropout(x1)
    x1a = F.relu(x1)
    x2 = self.linear2(x1a)
    output = self.output(x2)
    return output.view(-1)
