@staticmethod
def _parse_pipeline_type(pipeline_type):
    pipe = pipeline_type.lower()
    assert pipe in ('train', 'val'), 'Invalid pipeline type ("train", "val").'
    return pipe
