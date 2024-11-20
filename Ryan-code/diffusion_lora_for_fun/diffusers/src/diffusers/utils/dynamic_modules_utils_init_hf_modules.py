def init_hf_modules():
    """
    Creates the cache directory for modules with an init, and adds it to the Python path.
    """
    if HF_MODULES_CACHE in sys.path:
        return
    sys.path.append(HF_MODULES_CACHE)
    os.makedirs(HF_MODULES_CACHE, exist_ok=True)
    init_path = Path(HF_MODULES_CACHE) / '__init__.py'
    if not init_path.exists():
        init_path.touch()
