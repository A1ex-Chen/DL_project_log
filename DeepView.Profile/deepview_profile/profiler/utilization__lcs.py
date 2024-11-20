def _lcs(self, forward, backward):
    N, M = len(forward), len(backward)
    dp = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dp[i, j] = int(self._can_match(forward[i].name, backward[j][
                'name']))
            if i > 0:
                dp[i, j] = max(dp[i, j], dp[i - 1, j])
            if j > 0:
                dp[i, j] = max(dp[i, j], dp[i, j - 1])
            if i > 0 and j > 0 and self._can_match(forward[i].name,
                backward[j]['name']):
                dp[i, j] = max(dp[i, j], 1 + dp[i - 1, j - 1])
    matchings = []
    i, j = N - 1, M - 1
    while i > 0 and j > 0:
        if self._can_match(forward[i].name, backward[j]['name']):
            matchings.append((forward[i], backward[j]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i - 1, j] == dp[i, j]:
            i -= 1
        else:
            j -= 1
    if logging.root.level == logging.DEBUG:
        logger.debug(f'N = {N}, M = {M}, best matching: {len(matchings)}\n')
    return matchings
