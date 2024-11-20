def helper_computeIntersection(cp1: torch.Tensor, cp2: torch.Tensor, s:
    torch.Tensor, e: torch.Tensor):
    dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
    dp = [s[0] - e[0], s[1] - e[1]]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return torch.stack([(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 *
        dc[1]) * n3])
