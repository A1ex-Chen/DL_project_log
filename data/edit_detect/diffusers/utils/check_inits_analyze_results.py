def analyze_results(import_dict_objects, type_hint_objects):
    """
    Analyze the differences between _import_structure objects and TYPE_CHECKING objects found in an init.
    """

    def find_duplicates(seq):
        return [k for k, v in collections.Counter(seq).items() if v > 1]
    if list(import_dict_objects.keys()) != list(type_hint_objects.keys()):
        return ['Both sides of the init do not have the same backends!']
    errors = []
    for key in import_dict_objects.keys():
        duplicate_imports = find_duplicates(import_dict_objects[key])
        if duplicate_imports:
            errors.append(
                f'Duplicate _import_structure definitions for: {duplicate_imports}'
                )
        duplicate_type_hints = find_duplicates(type_hint_objects[key])
        if duplicate_type_hints:
            errors.append(
                f'Duplicate TYPE_CHECKING objects for: {duplicate_type_hints}')
        if sorted(set(import_dict_objects[key])) != sorted(set(
            type_hint_objects[key])):
            name = 'base imports' if key == 'none' else f'{key} backend'
            errors.append(f'Differences for {name}:')
            for a in type_hint_objects[key]:
                if a not in import_dict_objects[key]:
                    errors.append(
                        f'  {a} in TYPE_HINT but not in _import_structure.')
            for a in import_dict_objects[key]:
                if a not in type_hint_objects[key]:
                    errors.append(
                        f'  {a} in _import_structure but not in TYPE_HINT.')
    return errors
