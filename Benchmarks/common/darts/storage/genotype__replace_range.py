def _replace_range(self, genotype: Genotype) ->Genotype:
    """Replace the range values with lists

        Python's `range` is not serializable as json objects.
        We convert the genotype's ranges to lists first.

        Args:
            genotype: the genotype to be serialized

        Returns
            genotype: with proper lists.
        """
    genotype = genotype._replace(normal_concat=list(genotype.normal_concat))
    genotype = genotype._replace(reduce_concat=list(genotype.reduce_concat))
    return genotype
