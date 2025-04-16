def area(self):
    """
        Computes area of the mask.
        Only works with Polygons, using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Returns:
            Tensor: a vector, area for each instance
        """
    area = []
    for polygons_per_instance in self.polygons:
        area_per_instance = 0
        for p in polygons_per_instance:
            area_per_instance += polygon_area(p[0::2], p[1::2])
        area.append(area_per_instance)
    return torch.tensor(area)
