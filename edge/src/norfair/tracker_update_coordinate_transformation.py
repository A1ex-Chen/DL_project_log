def update_coordinate_transformation(self, coordinate_transformation:
    CoordinatesTransformation):
    if coordinate_transformation is not None:
        self.absolute_points = coordinate_transformation.rel_to_abs(self.
            absolute_points)
