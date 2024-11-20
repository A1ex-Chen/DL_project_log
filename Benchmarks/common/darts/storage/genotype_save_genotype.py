def save_genotype(self, genotype: Genotype, filename='genotype.json') ->None:
    """Save a genotype to disk

        Args:
            genotype: genotype to be saved
            filename: name of the save file
        """
    genotype = self._replace_range(genotype)
    os.makedirs(self.root, exist_ok=True)
    path = os.path.join(self.root, filename)
    with open(path, 'w') as outfile:
        json.dump(genotype, outfile)
