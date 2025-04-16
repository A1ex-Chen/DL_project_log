def load_genotype(self, filename='genotype.json') ->Genotype:
    """Load a genotype from disk

        Args:
            filename: name of the save file

        Returns:
            the genotype
        """
    path = os.path.join(self.root, filename)
    with open(path, 'r') as infile:
        saved = json.load(infile)
    genotype = self._convert_serialized(saved)
    return genotype
