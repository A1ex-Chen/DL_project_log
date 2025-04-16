def test_pydantic_error_truncation():


    class ListModel(pydantic.BaseModel):
        values: List[int]
    with pytest.raises(ModelkitDataValidationException):
        try:
            ListModel(values=['ok'] * 100)
        except pydantic.ValidationError as exc:
            raise ModelkitDataValidationException('test error',
                pydantic_exc=exc) from exc
    with pytest.raises(ModelkitDataValidationException):
        try:
            ListModel(values=['ok'])
        except pydantic.ValidationError as exc:
            raise ModelkitDataValidationException('test error',
                pydantic_exc=exc) from exc
