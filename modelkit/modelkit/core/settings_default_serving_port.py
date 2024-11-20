@pydantic.field_validator('port')
@classmethod
def default_serving_port(cls, v, values):
    if not v:
        v = 8500 if values.get('mode') == 'grpc' else 8501
    return v
