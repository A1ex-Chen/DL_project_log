def box_parametrization_to_corners(self, box_center_unnorm, box_size_unnorm,
    box_angle):
    return self.dataset_config.box_parametrization_to_corners(box_center_unnorm
        , box_size_unnorm, box_angle)
