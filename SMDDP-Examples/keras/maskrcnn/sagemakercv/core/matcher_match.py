def match(self, similarity_matrix, scope=None, **params):
    """Computes matches among row and column indices and returns the result.

    Computes matches among the row and column indices based on the similarity
    matrix and optional arguments.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher value means more similar.
      scope: Op scope name. Defaults to 'Match' if None.
      **params: Additional keyword arguments for specific implementations of
        the Matcher.

    Returns:
      A Match object with the results of matching.
    """
    return Match(self._match(similarity_matrix, **params))
