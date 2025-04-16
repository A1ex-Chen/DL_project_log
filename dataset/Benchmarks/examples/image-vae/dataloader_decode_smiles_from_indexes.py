def decode_smiles_from_indexes(self, vec):
    return ''.join(map(lambda x: self.charset[x], vec)).strip()
