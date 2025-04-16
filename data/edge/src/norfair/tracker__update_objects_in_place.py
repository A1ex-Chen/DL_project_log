def _update_objects_in_place(self, distance_function, distance_threshold,
    objects: Sequence['TrackedObject'], candidates: Optional[Union[List[
    'Detection'], List['TrackedObject']]], period: int):
    if candidates is not None and len(candidates) > 0:
        distance_matrix = distance_function.get_distances(objects, candidates)
        if np.isnan(distance_matrix).any():
            print(
                """
Received nan values from distance function, please check your distance function for errors!"""
                )
            exit()
        if distance_matrix.any():
            for i, minimum in enumerate(distance_matrix.min(axis=0)):
                objects[i].current_min_distance = (minimum if minimum <
                    distance_threshold else None)
        matched_cand_indices, matched_obj_indices = self.match_dets_and_objs(
            distance_matrix, distance_threshold)
        if len(matched_cand_indices) > 0:
            unmatched_candidates = [d for i, d in enumerate(candidates) if 
                i not in matched_cand_indices]
            unmatched_objects = [d for i, d in enumerate(objects) if i not in
                matched_obj_indices]
            matched_objects = []
            for match_cand_idx, match_obj_idx in zip(matched_cand_indices,
                matched_obj_indices):
                match_distance = distance_matrix[match_cand_idx, match_obj_idx]
                matched_candidate = candidates[match_cand_idx]
                matched_object = objects[match_obj_idx]
                if match_distance < distance_threshold:
                    if isinstance(matched_candidate, Detection):
                        matched_object.hit(matched_candidate, period=period)
                        matched_object.last_distance = match_distance
                        matched_objects.append(matched_object)
                    elif isinstance(matched_candidate, TrackedObject):
                        matched_object.merge(matched_candidate)
                        self.tracked_objects.remove(matched_candidate)
                else:
                    unmatched_candidates.append(matched_candidate)
                    unmatched_objects.append(matched_object)
        else:
            unmatched_candidates, matched_objects, unmatched_objects = (
                candidates, [], objects)
    else:
        unmatched_candidates, matched_objects, unmatched_objects = [], [
            ], objects
    return unmatched_candidates, matched_objects, unmatched_objects
