def _compute_distance(self, stacked_candidates: np.ndarray, stacked_objects:
    np.ndarray) ->np.ndarray:
    """
        Method that computes the pairwise distances between new candidates and objects.
        It is intended to use the entire vectors to compare to each other in a single operation.

        Parameters
        ----------
        stacked_candidates : np.ndarray
            np.ndarray containing a stack of candidates to be compared with the stacked_objects.
        stacked_objects : np.ndarray
            np.ndarray containing a stack of objects to be compared with the stacked_objects.

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
    return self.distance_function(stacked_candidates, stacked_objects)
