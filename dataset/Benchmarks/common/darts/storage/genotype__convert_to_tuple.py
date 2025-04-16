def _convert_to_tuple(self, block: list) ->List[tuple]:
    """Convert list to list of tuples

        Used when converting part of a serialized form of
        the genotype

        Args:
            block: part of the serialized genotype

        Returns:
            list of tuples that constitute that block
        """
    return [tuple(x) for x in block]
