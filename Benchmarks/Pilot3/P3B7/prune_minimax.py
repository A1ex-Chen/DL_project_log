def minimax(self, depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    MIN = -1000
    MAX = 1000
    if depth == 3:
        return values[nodeIndex]
    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = self.minimax(depth + 1, nodeIndex * 2 + i, False, values,
                alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = self.minimax(depth + 1, nodeIndex * 2 + i, True, values,
                alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best
