@staticmethod
def jensen_shannon_divergence(p, q):
    """
        Calculates the symmetric Jensen–Shannon divergence between the two PDFs
        """
    m = (p + q) * 0.5
    return 0.5 * (NDB.kl_divergence(p, m) + NDB.kl_divergence(q, m))
