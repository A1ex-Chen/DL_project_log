def get_table_schema(vector_size):
    """Extracts and returns the schema of a database table."""
    from lancedb.pydantic import LanceModel, Vector


    class Schema(LanceModel):
        im_file: str
        labels: List[str]
        cls: List[int]
        bboxes: List[List[float]]
        masks: List[List[List[int]]]
        keypoints: List[List[List[float]]]
        vector: Vector(vector_size)
    return Schema
