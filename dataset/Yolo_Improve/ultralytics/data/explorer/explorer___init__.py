def __init__(self, data: Union[str, Path]='coco128.yaml', model: str=
    'yolov8n.pt', uri: str=USER_CONFIG_DIR / 'explorer') ->None:
    """Initializes the Explorer class with dataset path, model, and URI for database connection."""
    checks.check_requirements(['lancedb>=0.4.3', 'duckdb<=0.9.2'])
    import lancedb
    self.connection = lancedb.connect(uri)
    self.table_name = f'{Path(data).name.lower()}_{model.lower()}'
    self.sim_idx_base_name = f'{self.table_name}_sim_idx'.lower()
    self.model = YOLO(model)
    self.data = data
    self.choice_set = None
    self.table = None
    self.progress = 0
