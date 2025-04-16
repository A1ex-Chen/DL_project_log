@staticmethod
def parking_regions_extraction(json_file):
    """
        Extract parking regions from json file.

        Args:
            json_file (str): file that have all parking slot points
        """
    with open(json_file, 'r') as json_file:
        return json.load(json_file)
