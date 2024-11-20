def on_track_sequence_end(self, tracker: MultiObjectTracker) ->None:
    """Call the `vector_model.predict` method on each tracklet to classify it into a team ID.

        Method called at the end of a track sequence. During this phase, team classification
        is performed on each tracklet using the `vector_model.predict`.

        Args:
            tracker (MultiObjectTracker): The instance of the tracker.

        Notes:
            - Team classification is applied to each tracklet.
            - An N-dimensional feature vector is extracted for each tracklet
            using `tracklet.get_observations(“feature”)`.
            - `vector_model.predict` is used to classify the tracklet into a team ID
            (0 or 1 in a 2-class problem).
        """
    logger.debug('Applying team classification method...')
    all_tracklets = tracker.alive_tracklets + tracker.dead_tracklets
    for tracklet in all_tracklets:
        tracklet_features: Vector = tracklet.get_observations('feature')
        predicted_team_id = self.vector_model(tracklet_features)
        most_frequent_team_id = stats.mode(predicted_team_id, axis=0,
            keepdims=False).mode
        tracklet.team_id = most_frequent_team_id
