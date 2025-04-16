@pydantic.field_validator('model_dependencies', mode='before')
@classmethod
def validate_dependencies(cls, v):
    if v is None:
        return {}
    if isinstance(v, (list, set)):
        return {key: key for key in v}
    return v
