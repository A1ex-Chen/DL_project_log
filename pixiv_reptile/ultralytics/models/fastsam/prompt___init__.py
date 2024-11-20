def __init__(self, source, results, device='cuda') ->None:
    """Initializes FastSAMPrompt with given source, results and device, and assigns clip for linear assignment."""
    if isinstance(source, (str, Path)) and os.path.isdir(source):
        raise ValueError(
            'FastSAM only accepts image paths and PIL Image sources, not directories.'
            )
    self.device = device
    self.results = results
    self.source = source
    try:
        import clip
    except ImportError:
        checks.check_requirements('git+https://github.com/ultralytics/CLIP.git'
            )
        import clip
    self.clip = clip
