def forward(self, query, keys):
    """

        :param query: if batch_first: (b x t_q x n) else: (t_q x b x n)
        :param keys: if batch_first: (b x t_k x n) else (t_k x b x n)

        :returns: (context, scores_normalized)
        context: if batch_first: (b x t_q x n) else (t_q x b x n)
        scores_normalized: if batch_first (b x t_q x t_k) else (t_q x b x t_k)
        """
    if not self.batch_first:
        keys = keys.transpose(0, 1)
        if query.dim() == 3:
            query = query.transpose(0, 1)
    if query.dim() == 2:
        single_query = True
        query = query.unsqueeze(1)
    else:
        single_query = False
    b = query.size(0)
    t_k = keys.size(1)
    t_q = query.size(1)
    processed_query = self.linear_q(query)
    processed_key = self.linear_k(keys)
    scores = self.calc_score(processed_query, processed_key)
    if self.mask is not None:
        mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
        scores.data.masked_fill_(mask, -65504.0)
    scores_normalized = F.softmax(scores, dim=-1)
    context = torch.bmm(scores_normalized, keys)
    if single_query:
        context = context.squeeze(1)
        scores_normalized = scores_normalized.squeeze(1)
    elif not self.batch_first:
        context = context.transpose(0, 1)
        scores_normalized = scores_normalized.transpose(0, 1)
    return context, scores_normalized
