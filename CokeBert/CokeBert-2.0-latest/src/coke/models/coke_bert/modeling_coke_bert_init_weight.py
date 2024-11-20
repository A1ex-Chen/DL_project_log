def init_weight(self):
    for module in [self.K_V_linear_1, self.K_V_linear_2, self.V_linear_1,
        self.V_linear_2, self.Q_linear_1, self.Q_linear_2, self.comb]:
        init.xavier_uniform_(module.weight)
    if self.config.graphsage:
        init.xavier_uniform_(self.graphsage_linear.weight)
