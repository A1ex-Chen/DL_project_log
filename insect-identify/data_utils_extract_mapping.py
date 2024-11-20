def extract_mapping(mapping_file_path, classes):
    """ Build a mapping dictionnary from a json mapping file
        Build a most likely species names from provided classes
        return the species
    """
    with open(mapping_file_path, 'r') as f:
        cat_to_name = json.load(f)
    species = []
    for i in classes:
        species += [cat_to_name[i]]
    return species
