def create_dynamic_module(name: Union[str, os.PathLike]):
    """
    Creates a dynamic module in the cache directory for modules.
    """
    init_hf_modules()
    dynamic_module_path = Path(HF_MODULES_CACHE) / name
    if not dynamic_module_path.parent.exists():
        create_dynamic_module(dynamic_module_path.parent)
    os.makedirs(dynamic_module_path, exist_ok=True)
    init_path = dynamic_module_path / '__init__.py'
    if not init_path.exists():
        init_path.touch()
