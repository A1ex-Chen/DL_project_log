def calc_score(self, att_query, att_keys):
    """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        returns: b x t_q x t_k scores
        """
    b, t_k, n = att_keys.size()
    t_q = att_query.size(1)
    att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
    att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
    sum_qk = att_query + att_keys
    if self.normalize:
        sum_qk = sum_qk + self.normalize_bias
        linear_att = self.linear_att / self.linear_att.norm()
        linear_att = linear_att * self.normalize_scalar
    else:
        linear_att = self.linear_att
    out = torch.tanh(sum_qk).matmul(linear_att)
    return out
