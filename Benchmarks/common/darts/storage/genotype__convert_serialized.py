def _convert_serialized(self, save: list) ->Genotype:
    """Convert json serialized form to Genotype

        Args:
            save: serialized form of the the genotype

        Returns:
            the genotype
        """
    normal = self._convert_to_tuple(save[0])
    normal_concat = save[1]
    reduce = self._convert_to_tuple(save[2])
    reduce_concat = save[3]
    return Genotype(normal, normal_concat, reduce, reduce_concat)
