def init_weights(self):
    nn.init.xavier_uniform_(self.fc_q.weight)
    nn.init.xavier_uniform_(self.fc_k.weight)
    nn.init.xavier_uniform_(self.fc_v.weight)
    nn.init.xavier_uniform_(self.fc_o.weight)
    nn.init.constant_(self.fc_q.bias, 0)
    nn.init.constant_(self.fc_k.bias, 0)
    nn.init.constant_(self.fc_v.bias, 0)
    nn.init.constant_(self.fc_o.bias, 0)
