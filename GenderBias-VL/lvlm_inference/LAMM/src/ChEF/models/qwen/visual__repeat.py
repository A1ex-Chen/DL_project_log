def _repeat(self, query, N: int):
    return query.unsqueeze(1).repeat(1, N, 1)
