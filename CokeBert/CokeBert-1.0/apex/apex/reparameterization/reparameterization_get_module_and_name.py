@staticmethod
def get_module_and_name(module, name):
    """
        recursively fetches (possible) child module and name of weight to be reparameterized
        """
    name2use = None
    module2use = None
    names = name.split('.')
    if len(names) == 1 and names[0] != '':
        name2use = names[0]
        module2use = module
    elif len(names) > 1:
        module2use = module
        name2use = names[0]
        for i in range(len(names) - 1):
            module2use = getattr(module2use, name2use)
            name2use = names[i + 1]
    return module2use, name2use
