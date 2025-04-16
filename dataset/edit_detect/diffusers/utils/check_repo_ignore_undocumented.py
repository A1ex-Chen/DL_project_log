def ignore_undocumented(name):
    """Rules to determine if `name` should be undocumented."""
    if name.isupper():
        return True
    if name.endswith('ModelMixin') or name.endswith('Decoder'
        ) or name.endswith('Encoder') or name.endswith('Layer'
        ) or name.endswith('Embeddings') or name.endswith('Attention'):
        return True
    if os.path.isdir(os.path.join(PATH_TO_DIFFUSERS, name)) or os.path.isfile(
        os.path.join(PATH_TO_DIFFUSERS, f'{name}.py')):
        return True
    if name.startswith('load_tf') or name.startswith('load_pytorch'):
        return True
    if name.startswith('is_') and name.endswith('_available'):
        return True
    if name in DEPRECATED_OBJECTS or name in UNDOCUMENTED_OBJECTS:
        return True
    if name.startswith('MMBT'):
        return True
    if name in SHOULD_HAVE_THEIR_OWN_PAGE:
        return True
    return False
