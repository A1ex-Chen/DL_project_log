def create(self, initial_detection: 'Detection', hit_counter_max: int,
    initialization_delay: int, pointwise_hit_counter_max: int,
    detection_threshold: float, period: int, filter_factory:
    'FilterFactory', past_detections_length: int, reid_hit_counter_max:
    Optional[int], coord_transformations: CoordinatesTransformation
    ) ->'TrackedObject':
    obj = TrackedObject(obj_factory=self, initial_detection=
        initial_detection, hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
        pointwise_hit_counter_max=pointwise_hit_counter_max,
        detection_threshold=detection_threshold, period=period,
        filter_factory=filter_factory, past_detections_length=
        past_detections_length, reid_hit_counter_max=reid_hit_counter_max,
        coord_transformations=coord_transformations)
    return obj
