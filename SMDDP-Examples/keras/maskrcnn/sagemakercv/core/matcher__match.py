@abstractmethod
def _match(self, similarity_matrix, **params):
    """Method to be overridden by implementations.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher value means more similar.
      **params: Additional keyword arguments for specific implementations of
        the Matcher.

    Returns:
      match_results: Integer tensor of shape [M]: match_results[i]>=0 means
        that column i is matched to row match_results[i], match_results[i]=-1
        means that the column is not matched. match_results[i]=-2 means that
        the column is ignored (usually this happens when there is a very weak
        match which one neither wants as positive nor negative example).
    """
    pass
