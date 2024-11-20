def self_attention(self, Q, K_1, V_1, K_2, V_2, mode=None):
    Q_2 = self.Q_linear_2(Q)
    Q_2 = self.Tanh(Q_2)
    Q_2 = Q_2.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    if self.config.neighbor_hop == 2:
        K = self.K_V_linear_2(K_2)
        attention = (Q_2 * K).sum(4).div(math.sqrt(self.config.K_V_dim))
        attention = attention.masked_fill(attention == 0, float('-10000'))
        attention = self.softmax_dim_3(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention == float(1 / attention.
            shape[-1]), float(0))
        attention = attention.unsqueeze(3)
        sentence_entity_reps = attention.matmul(V_2).squeeze(3)
        if mode == 'candidate_pos':
            V_mask = V_1.sum(3)
            V_mask = V_mask.masked_fill(V_mask != 0, float(1)).unsqueeze(-1)
            sentence_entity_reps = V_mask * sentence_entity_reps
            V_1 = V_mask * V_1
            V_1 = torch.cat([V_1, sentence_entity_reps], -1)
        elif mode == 'candidate_neg':
            V_mask = V_1.sum(3)
            V_mask = (V_mask.masked_fill(V_mask != 0, float(1)).unsqueeze(-
                1) - 1) * -1
            sentence_entity_reps = V_mask * sentence_entity_reps
            V_1 = V_mask * V_1
            V_1 = torch.cat([V_1, sentence_entity_reps], -1)
        else:
            V_1 = torch.cat([V_1, sentence_entity_reps], -1)
        Q_1 = self.Q_linear_1(Q)
        Q_1 = self.Tanh(Q_1)
        Q_1 = Q_1.unsqueeze(1).unsqueeze(2)
        K = self.K_V_linear_1(K_1)
        attention = (Q_1 * K).sum(3).div(math.sqrt(self.config.K_V_dim))
        attention = attention.masked_fill(attention == 0, float('-10000'))
        attention = self.softmax_dim_2(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention == float(1 / attention.
            shape[-1]), float(0))
        attention = attention.unsqueeze(2)
        sentence_entity_reps = attention.matmul(V_1).squeeze(2)
        return sentence_entity_reps
    elif self.config.neighbor_hop == 1:
        if mode == 'candidate_pos':
            V_mask = V_1.sum(3)
            V_mask = V_mask.masked_fill(V_mask != 0, float(1)).unsqueeze(-1)
            V_1 = V_mask * V_1
        elif mode == 'candidate_neg':
            V_mask = V_1.sum(3)
            V_mask = (V_mask.masked_fill(V_mask != 0, float(1)).unsqueeze(-
                1) - 1) * -1
            V_1 = V_mask * V_1
        else:
            V_1 = V_1
        Q_1 = self.Q_linear_1(Q)
        Q_1 = self.Tanh(Q_1)
        Q_1 = Q_1.unsqueeze(1).unsqueeze(2)
        K = self.K_V_linear_1(K_1)
        attention = (Q_1 * K).sum(3).div(math.sqrt(self.config.K_V_dim))
        attention = attention.masked_fill(attention == 0, float('-10000'))
        attention = self.softmax_dim_2(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention == float(1 / attention.
            shape[-1]), float(0))
        attention = attention.unsqueeze(2)
        sentence_entity_reps = attention.matmul(V_1).squeeze(2)
        return sentence_entity_reps
    else:
        raise NotImplementedError
