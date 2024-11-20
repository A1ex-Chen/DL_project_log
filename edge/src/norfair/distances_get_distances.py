def get_distances(self, objects: Sequence['TrackedObject'], candidates:
    Optional[Union[List['Detection'], List['TrackedObject']]]) ->np.ndarray:
    """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
    distance_matrix = np.full((len(candidates), len(objects)), fill_value=
        np.inf, dtype=np.float32)
    if not objects or not candidates:
        return distance_matrix
    object_labels = np.array([o.label for o in objects]).astype(str)
    candidate_labels = np.array([c.label for c in candidates]).astype(str)
    for label in np.intersect1d(np.unique(object_labels), np.unique(
        candidate_labels)):
        obj_mask = object_labels == label
        cand_mask = candidate_labels == label
        stacked_objects = []
        for o in objects:
            if str(o.label) == label:
                stacked_objects.append(o.estimate.ravel())
        stacked_objects = np.stack(stacked_objects)
        stacked_candidates = []
        for c in candidates:
            if str(c.label) == label:
                if 'Detection' in str(type(c)):
                    stacked_candidates.append(c.points.ravel())
                else:
                    stacked_candidates.append(c.estimate.ravel())
        stacked_candidates = np.stack(stacked_candidates)
        distance_matrix[np.ix_(cand_mask, obj_mask)] = self._compute_distance(
            stacked_candidates, stacked_objects)
    return distance_matrix
